"""
2075. Top-P采样
难度：中等

【题目描述】
在大语言模型的自回归文本生成中，每一步都会输出整个词表的 logits 得分。
传统的贪心解码、束搜索等最大化式解码方法，会导致生成文本出现重复、语义平淡、陷入循环的神经文本退化问题。

Top-P 采样又称核采样（Nucleus Sampling），出自《The Curious Case of Neural Text Degeneration》，
是对 Top-K 采样的自适应改进：
不再固定保留 K 个候选，而是按概率降序累加，保留概率质量总和达到并超过阈值 top_p 的最小候选集合（核），
其余位置的 logits 置为负无穷（对应概率为 0），再对截断后的 logits 重新计算 Softmax 得到最终采样分布。

该方法可以根据词表分布的扁平程度自适应调整候选数量：在低熵场景收缩候选集，在高熵场景扩展候选集，
更灵活地平衡生成质量与多样性。

【实现要求】
- 先对 logits 做 Softmax 得排序依据，再掩码重归一化。
- 保留 4 位小数。
- 概率值相同时，按原始下标升序优先保留。

【算法流程】
1. p = softmax(logits)
2. 按概率降序排序，寻找满足累积概率 sum >= top_p 的最小候选集合前缀。
3. 将未选中的位置对应的 logits 掩码置为 -inf。
4. 对掩码后的 logits 重新计算 softmax，并保留 4 位小数返回。

【参数说明】
- logits: 词表 logits (List[float])
- top_p: 核概率阈值 (float, 0 < top_p <= 1)

【返回值】
- 采样分布：处理后的概率向量（浮点比较绝对误差或相对误差 <= 10^-4）。

【限制条件】
- 向量长度 1 <= n <= 词表大小
- 0 < top_p <= 1
"""
import torch
def top_p_sample(logits:torch.Tensor,top_p:float)->torch.Tensor:
    logits = torch.as_tensor(logits,dtype = torch.float32)
    probs = torch.softmax(logits,dim=-1)
    #按照概率降序排列
    sorted_probs,sorted_indices = torch.sort(probs,descending=True,stable=True)
    #计算累积概率前缀和
    cumsum = torch.cumsum(sorted_probs,dim=-1)
    #寻找满足sum>=top_p的额最小候选前缀长度
    #cumsum<top_p统计的是还没达到阈值的索引数量，再+1就是刚好达成或超过的那一个
    #cutoff刚好保存保留的索引数量
    cutoff = min(torch.sum(cumsum<top_p).item()+1,len(logits))
    #保留的索引 前cutoff个 [0,超过/刚好达到top_p的索引]
    keep_idx = sorted_indices[:cutoff]
    masked = torch.full_like(logits,-float('inf'))
    masked[keep_idx] =logits[keep_idx]
    final_probs = torch.softmax(masked,dim=-1)
    return torch.round(final_probs,decimals=4)


if __name__ == '__main__':
    print("=" * 60)
    print("测试用例 1: 典型多候选截断 (top_p = 0.88)")
    logits1 = [1.0, 5.0, 2.0, 3.0]
    p1 = 0.88
    res1 = top_p_sample(logits1, p1)
    print(f"输入: logits = {logits1}, top_p = {p1}")
    print(f"输出采样分布: {res1}")

    print("\n" + "=" * 60)
    print("测试用例 2: 低熵极尖锐场景 (top_p = 0.5，首词即达标，自动收缩为 1 个词)")
    p2 = 0.5
    res2 = top_p_sample(logits1, p2)
    print(f"输入: logits = {logits1}, top_p = {p2}")
    print(f"输出采样分布: {res2}")

    print("\n" + "=" * 60)
    print("测试用例 3: 平局决胜 (分值相同时，按原始下标升序优先保留)")
    logits3 = [2.0, 2.0, 1.0]
    p3 = 0.4
    res3 = top_p_sample(logits3, p3)
    print(f"输入: logits = {logits3}, top_p = {p3}")
    print(f"输出采样分布: {res3}")

    print("\n" + "=" * 60)
    print("测试用例 4: 全词表保留 (top_p = 1.0)")
    p4 = 1.0
    res4 = top_p_sample(logits1, p4)
    print(f"输入: logits = {logits1}, top_p = {p4}")
    print(f"输出采样分布: {res4}")
    print("=" * 60)



