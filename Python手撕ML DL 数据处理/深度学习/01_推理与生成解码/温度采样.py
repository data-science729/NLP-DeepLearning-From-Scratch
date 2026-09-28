"""
题目：2073. 温度采样
难度：中等

【题目描述】
计算温度采样分布：对输入的 logits 进行温度 T 缩放后应用 Softmax，并返回各类别对应的概率分布（仅计算概率，不执行实际采样操作）。

【数学公式】
对于输入的每个类别 i，其概率 p_i 计算公式如下：
    p_i = exp(z_i / T) / sum_j(exp(z_j / T))

【实现要求】
1. 温度 T > 0。
2. 输出结果保留 4 位小数。

【参数说明】
- logits (List[float]): 词表原始未归一化的对数几率 (logits)。
- temperature (float): 采样温度参数 T。

【返回值】
- List[float]: 经过温度缩放与 Softmax 归一化后的概率分布列表。

【输出与判定】
- 返回浮点数列表。
- 浮点数判定标准：满足绝对误差 <= 1e-4 或相对误差 <= 1e-4 之一即可通过。

【示例 1】
输入：
    logits = [1, 2, 3]
    temperature = 1.0
输出：
    [0.09, 0.2447, 0.6652]

【限制条件】
- temperature > 0
"""
import torch
def temperature_sample(logits:torch.Tensor,temperature:float)->torch.Tensor:
    logits = torch.as_tensor(logits,dtype=torch.float32)
    t_scaled_logits = logits/temperature
    probs = torch.softmax(t_scaled_logits,dim=-1)
    return torch.round(probs,decimals=4)

if __name__ == '__main__':
    print("=" * 60)
    print("测试用例 1: 标准温度 T = 1.0 (题目示例 1)")
    logits1 = [1, 2, 3]
    t1 = 1.0
    res1 = temperature_sample(logits1, t1)
    print(f"输入: logits = {logits1}, T = {t1}")
    print(f"输出概率: {res1}")

    print("\n" + "=" * 60)
    print("测试用例 2: 低温环境 T = 0.5 (贫富差距拉大，高分项更占绝对优势)")
    t2 = 0.5
    res2 = temperature_sample(logits1, t2)
    print(f"输入: logits = {logits1}, T = {t2}")
    print(f"输出概率: {res2}")

    print("\n" + "=" * 60)
    print("测试用例 3: 高温环境 T = 2.0 (打土豪分田地，概率趋向平缓)")
    t3 = 2.0
    res3 = temperature_sample(logits1, t3)
    print(f"输入: logits = {logits1}, T = {t3}")
    print(f"输出概率: {res3}")

    print("\n" + "=" * 60)
    print("测试用例 4: 极度低温 T = 0.05 (逼近贪心搜索，第一名概率接近 100%)")
    t4 = 0.05
    res4 = temperature_sample(logits1, t4)
    print(f"输入: logits = {logits1}, T = {t4}")
    print(f"输出概率: {res4}")
    print("=" * 60)
