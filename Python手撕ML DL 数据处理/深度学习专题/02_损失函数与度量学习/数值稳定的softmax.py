"""
1083. Log-Softmax
难度：简单

【题目描述】
Softmax 将向量归一为概率分布：
    softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
对其取对数得到 log-softmax，常用于交叉熵损失的目标值。

如果直接按定义计算，当 x 包含较大元素（如 1000 量级），exp(x) 会溢出为 inf，
最终结果是 nan。

经典处理做法是同时减去最大值 m = max_i(x_i)：
    log_softmax(x)_i = x_i - m - ln(sum_j(exp(x_j - m)))
由于 x_j - m <= 0，指数运算不会溢出；减去的 m 在差里自动抵消，数学上与原定义等价。

本题给定一维得分向量，返回其 log-softmax。

【实现要求】
- 用 x - max(x) 后再做指数，避免溢出。
- 返回与输入等长的对数概率向量。

【参数说明】
- scores: 一维得分向量 (List[float])

【返回值】
- 结果：同形的浮点向量，浮点比较误差要求：绝对误差 <= 10^-5 或相对误差 <= 10^-5。

【限制条件】
- 向量长度 1 <= n <= 100
- 各分量 |x_i| <= 10^3
"""
import torch
import torch.nn.functional as F
#数值稳定的softmax 底层F.log_softmax

def log_softmax(scores: torch.Tensor) -> torch.Tensor:
    m = torch.max(scores)
    shifted = scores - m
    lse = torch.log(torch.sum(torch.exp(shifted)))
    return shifted - lse



