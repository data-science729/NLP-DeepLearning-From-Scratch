"""
3225. 温度Top-K与Top-P过滤 (中等)

题目描述：
    大模型解码生成时，通常将温度缩放（Temperature）、Top-K 截断与 Top-P（Nucleus）过滤结合使用：
    1. 先将 logits 除以温度 temperature 进行缩放。
    2. 执行 Top-K 过滤：只保留最大的 K 个值（top_k <= 0 表示跳过）。
    3. 执行 Top-P 过滤：对经 Top-K 处理后的 logits 按降序计算累积 Softmax 概率，
       截取累积概率达到 top_p 的最短前缀（top_p >= 1 表示跳过）。
    4. 被丢弃的位置置为 -inf（或在概率分布中对应为 0）。
    本题仅需返回过滤后的 logits 或其 Softmax 概率分布，不需要进行多项式采样。

实现要求：
    - 输入：一维或末维为词表大小（V）的 logits 数组/张量；返回同形状张量/数组。
    - Top-K（阈值法）：保留所有 >= 第 K 大值的元素（若存在数值并列，并列项全部保留）。
    - Top-P：对已完成 Top-K 的 logits 按降序排序并做 Softmax，取累积概率首次达到 top_p 的最短前缀；
      至少保留最高概率的 1 个元素。
    - 恢复原始顺序：将筛选结果映射回原索引顺序，未保留位置设为 -inf。

算法伪代码：
    x = logits / temperature
    if top_k > 0:
        thr = kth_largest(x, top_k)
        x[x < thr] = -inf  # 并列等于 thr 的元素全部保留
    if top_p < 1.0:
        sort x descending -> softmax -> cumsum
        keep shortest prefix with cumsum >= top_p (at least 1)
        scatter back to original index order, non-selected = -inf
    return x

参数说明：
    logits (list[float] | np.ndarray | torch.Tensor): 未归一化打分，末维大小为词表 V
    temperature (float): 温度系数，默认值为 1.0 (temperature > 0)
    top_k (int): 保留的前 K 大个数，<= 0 表示关闭 Top-K 过滤
    top_p (float): 累积概率阈值，>= 1.0 表示关闭 Top-P 过滤 (0 < top_p <= 1.0)

返回值：
    同输入维度的数组/张量：返回过滤后的 logits（被过滤位置为 -inf）或 Softmax 归一化后的概率分布。

评测判定：
    - 浮点比较满足绝对误差 <= 1e-4 或相对误差 <= 1e-4 即视为通过。
    - logits 中的 -inf 位置在概率对比时按 0 处理。

示例 1:
    输入:
        logits = [1, 2, 3, 4]
        temperature = 1.0
        top_k = 2
        top_p = 1.0
    输出:
        返回值 = [0.0, 0.0, 0.26894142136999516, 0.7310585786300049]
        (或 logits 形式: [-inf, -inf, 3.0, 4.0])

限制条件:
    - logits 末维为词表维度 V <= 32
    - temperature > 0
    - top_k >= 0
    - 0 < top_p <= 1.0
    - 不涉及随机多项式采样
"""
import torch
def top_k_top_p_filter(logits:torch.Tensor,temperature:float,top_k:int,top_p:float)->torch.Tensor:
    #确保为浮点张量并clone一份 避免就地修改破坏外部变量
    x = torch.as_tensor(logits,dtype =torch.float32).clone()
    x = x / temperature
    #阶段2：Top-k过滤(top_k<=0表示跳过) x.shape[-1]词表大小
    #粗筛 阈值保留法
    if top_k >0 and top_k <x.shape[-1]:
        #...是一个合格的内置语法叫Ellipsis(省略号) 表示前面的维度全当:,[...,-1]表示只切片最后一个维度
        #当然对于本题的x 维度就一个 values[-1]和values[...,-1]完全一样
        #但多维张量如(batch_size,vocab_size)
        thr = torch.topk(x,k=top_k,dim=-1).values[...,-1]
        x[x<thr] = -torch.inf
    #阶段三:Top-p过滤(核采样)
    if top_p<1.0:
        sorted_logits,sorted_idx = torch.sort(x,descending=True,stable=True)
        sorted_probs = torch.softmax(sorted_logits,dim=-1)
        cumsum = torch.cumsum(sorted_probs,dim=-1)
        cutoff = min(torch.sum(cumsum<top_p).item()+1,len(logits))
        keep_idx = sorted_idx[:cutoff]
        new_x = torch.full_like(x,-torch.inf)
        new_x[keep_idx] = x[keep_idx]
        x = new_x
    return x

if __name__ == '__main__':
    print("=" * 65)
    print("测试用例 1: 题目示例 1 (Top-K=2, Top-P=1.0 跳过)")
    l1 = [1, 2, 3, 4]
    res1 = top_k_top_p_filter(l1, temperature=1.0, top_k=2, top_p=1.0)
    print(f"输入: logits = {l1}, T = 1.0, top_k = 2, top_p = 1.0")
    print(f"输出过滤后的 Logits: {res1}")
    print(f"对应 Softmax 概率分布: {torch.round(torch.softmax(res1, dim=-1), decimals=4)}")

    print("\n" + "=" * 65)
    print("测试用例 2: 阈值并列项全部保留 (分值并列时，并列项皆 >= thr 保留)")
    l2 = [1, 3, 3, 4]
    res2 = top_k_top_p_filter(l2, temperature=1.0, top_k=2, top_p=1.0)
    print(f"输入: logits = {l2}, T = 1.0, top_k = 2, top_p = 1.0")
    print(f"输出过滤后的 Logits: {res2}")
    print(f"对应 Softmax 概率分布: {torch.round(torch.softmax(res2, dim=-1), decimals=4)}")

    print("\n" + "=" * 65)
    print("测试用例 3: 终极组合拳 (温度 T=0.5 缩放 + Top-K=3 粗筛 + Top-P=0.8 细筛)")
    l3 = [1, 2, 3, 4]
    res3 = top_k_top_p_filter(l3, temperature=0.5, top_k=3, top_p=0.8)
    print(f"输入: logits = {l3}, T = 0.5, top_k = 3, top_p = 0.8")
    print(f"输出过滤后的 Logits: {res3}")
    print(f"对应 Softmax 概率分布: {torch.round(torch.softmax(res3, dim=-1), decimals=4)}")

    print("\n" + "=" * 65)
    print("测试用例 4: 默认全通过 (top_k=0, top_p=1.0，仅做温度缩放)")
    l4 = [1, 2, 3]
    res4 = top_k_top_p_filter(l4, temperature=2.0, top_k=0, top_p=1.0)
    print(f"输入: logits = {l4}, T = 2.0, top_k = 0, top_p = 1.0")
    print(f"输出过滤后的 Logits: {res4}")
    print("=" * 65)



