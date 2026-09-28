"""
2072. 束搜索解码
难度：中等

【题目描述】
对逐步给出的词表 logits 做束搜索：每步保留 beam_width 条候选，用对数概率累加扩展，长度惩罚后取最优序列。

【实现要求】
- logits_steps 长度 T，每项为词表维 V 的原始 logits；第 t 步所有束共用 logits_steps[t]。
- length_penalty 默认 alpha = 1；eos_id 默认 None。
- 已以 EOS 结尾的束不再扩展，原样进入候选；再与其它候选一起按 score 取 top-beam_width（冻结束也可能被截掉）。
- 扩展用 sum(log p)，禁止概率连乘；每步截断与最终选择都用长度惩罚分 score(s)。
- 长度惩罚公式：
    score(s) = (sum_{t=1}^L log p_{y_t}) / (L^alpha)
    其中 L = max(|s|, 1) 包含已写入的 EOS。
    当 alpha = 0 时 L^0 = 1，不做长度归一；alpha 增大更利于较长序列（在 sum(log p) < 0 时）。
- 打分并列时取字典序更小的 token 序列（seq asc）。
- 返回最佳序列的 Python list[int]。

【算法流程】
1. 初始化 beams = [([], 0.0)]。
2. 对每步 logits 做 Softmax 得到 p，并计算对数概率 lp = log(p + 1e-12)。
3. 遍历当前 beams，对未以 EOS 结尾的束进行全词表扩展；对已结束的束直接保留。
4. 按照 score 降序、seq 升序进行排序，截取前 beam_width 个候选。
5. 循环结束后返回最优序列。

【参数说明】
- logits_steps: 逐步 logits [T, V] (List[List[float]])
- beam_width: 束宽 (int)
- length_penalty: 长度惩罚系数 alpha (float)
- eos_id: 结束符 ID (int 或 None)

【返回值】
- 结果：最优 token 序列 (List[int])

【限制条件】
- 1 <= T <= 16, 1 <= V <= 32, 1 <= beam_width <= 8
- 0 <= alpha <= 3；eos_id 为 None 或 0 <= eos_id < V
- 各步 logits 等长、有限浮点
"""
import torch
from typing import Optional




def beam_search_decode(
    logits_steps: torch.Tensor,
    beam_width: int,
    length_penalty: float = 1.0,
    eos_id: Optional[int] = None,
) -> list[int]:


    T,V = logits_steps.shape   #拿到总步数T 和词表大小V
    beams = [([],0.0)]   #初始化候选列表(序列，累计对数概率)
    #按时间步逐步解码
    for t in range(T):
        logits_t = logits_steps[t]
        p = torch.softmax(logits_t,dim = -1)  #为何dim=-1? 一维(V,)二维(Batch,V),三维(Batch,Seq,V）最后一维永远是词表维
        lp = torch.log(p+1e-12)
        candidates = []

    #遍历所有句子生成新可能
        for seq,cum_lp in beams:
            #防止空列表 防止没有结束符 句子最后一个词是结束符
            if eos_id is not None and len(seq)>0 and seq[-1] ==eos_id:
                L = max(len(seq),1)
                score = cum_lp/(L**length_penalty)
            #候选存放三元组(序列，原始累积对数分，惩罚后得分)
                candidates.append((seq,cum_lp,score))
            else:
                for v in range(V):
                    new_seq = seq +[v]
                    new_cum_lp = cum_lp+lp[v].item()
                    L = max(len(new_seq),1)
                    score = new_cum_lp/(L**length_penalty)
                    candidates.append((new_seq,new_cum_lp,score))

        # 自定义排序函数:三元组(seq,cum_lp,score) 这个函数实现了按score降序排序，score一样按词表字典序升序排序
        def get_sort_key(candidate):
            return (-candidate[2], candidate[0])

        candidates.sort(key = get_sort_key)
        new_beams = []
        for i in range(min(beam_width,len(candidates))):
            cand = candidates[i]
            new_beams.append((cand[0],cand[1]))
        beams = new_beams  # 👈【8个空格】与 for i 对齐！还在 for t 里面
    return beams[0][0]  # 👈【4个空格】与 for t 对齐！跳出整个大循环后交卷


if __name__ == "__main__":
    # 固定随机种子以便复现
    torch.manual_seed(42)
    # 模拟数据：4 个时间步，词表大小为 5
    sample_logits = torch.randn(4, 5)
    # 测试 1：没有结束符的常规测试
    best_seq1 = beam_search_decode(sample_logits, beam_width=3, length_penalty=1.0)
    print("最优序列 (无 EOS):", best_seq1)
    # 测试 2：假设词表里的 2 号是结束符 EOS
    best_seq2 = beam_search_decode(sample_logits, beam_width=3, length_penalty=1.0, eos_id=2)
    print("最优序列 (设 EOS=2):", best_seq2)























