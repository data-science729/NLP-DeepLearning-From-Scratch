"""
2088. KL散度
难度: 中等
标签: 企业A

【题目描述】
计算两个离散分布之间的 Kullback-Leibler 散度 D_KL(p || q)。
输入向量不一定已经归一化（样例碰巧和为 1）。计算前须先分别做 L1 归一化：
    p_hat_i = p_i / sum(p)
    q_hat_i = q_i / sum(q)
再对 p_hat, q_hat 计算 KL 散度，对数底数取自然对数 ln。

【计算公式】
    D_KL(p_hat || q_hat) = sum_{i} p_hat_i * ln(p_hat_i / q_hat_i)

【边界约定】
1. 0 * ln(0) = 0：当 p_hat_i == 0 时，该项对总和贡献为 0。
2. 异常项跳过：若 q_hat_i == 0 且 p_hat_i > 0，题目约定该分量按 0 计（跳过该项）。

【参数说明】
:param p: List[float]，目标分布未归一化或已归一化的向量
:param q: List[float]，近似分布未归一化或已归一化的向量
:return: float，计算所得的 KL 散度标量值

【判题标准】
返回 float 标量；判定条件为绝对误差 <= 1e-4 或相对误差 <= 1e-4。

【限制条件】
- 向量长度: 1 <= len(p) == len(q) <= 64
- p_i, q_i >= 0，且 sum(p) > 0, sum(q) > 0

【示例 1】
输入：
    p = [0.5, 0.5]
    q = [0.5, 0.5]
输出：
    0.0
"""
import torch

def kl_divergence(p:list[float],q:list[float])->float:
    p_tensor = torch.tensor(p,dtype = torch.float64)
    q_tensor = torch.tensor(q,dtype = torch.float64)
    #归一化  向量/标量 发生广播机制 pytorch自动把分母的单个数字 分别除给分子里的每一个元素
    p_hat = p_tensor / torch.sum(p_tensor)
    q_hat = q_tensor /torch.sum(q_tensor)
    #边界条件过滤 只有p_hat>0 且q_hat>0的分量才参与计算
    #mask 本质是由True和False组成的布尔型张量(Boolean Tensor) True表示我要这一项请保留并参与计算
    #False的部分表示我不要这一项 请过滤丢弃屏蔽
    mask = (p_hat>0) & (q_hat>0)  #位逻辑与运算符 element-wise 在tensor里边用 不能用and
    #如果没有任何有效项 返回0.0
    if not torch.any(mask):
        return 0.0

    p_valid = p_hat[mask]
    q_valid = q_hat[mask]
    kl = torch.sum(p_valid*torch.log(p_valid/q_valid))
    #调用item 转为python原生的float标量返回
    return float(kl.item())



