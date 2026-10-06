"""
2067. 多层感知机前向
难度: 中等

【题目描述】
两层感知机前向：隐层 ReLU，输出层线性，带偏置。

【实现要求】
- 矩阵乘法按 batch × 特征约定。
- 输出可压成一维向量（或保持二维 batch 形式，根据维度约定与评测判定返回兼容形状）。

【算法公式】
    h = ReLU(X · W1 + b1)
    Y = h · W2 + b2

【伪代码】
h = relu(X @ W1 + b1)
Y = h @ W2 + b2
return squeeze(Y)

【输出与判定】
返回 batch 输出；浮点比较：绝对误差 <= 10^-5 或相对误差 <= 10^-5（满足其一即通过）。

【参数说明】
- X: 输入 batch 矩阵
- W1: 第一层权重矩阵
- b1: 第一层偏置
- W2: 第二层权重矩阵
- b2: 第二层偏置

【返回值】
前向传播输出结果

【示例 1】
输入：
    X (2, 2) = [[1, 2], [3, 4]]
    W1 (2, 2) = [[1, 0], [0, 1]]
    b1 = [0, 0]
    W2 (2, 1) = [[1], [2]]
    b2 = 0
输出：
    返回值 (2, 1) = [[5.0], [11.0]]

【限制条件】
- batch 与特征维 <= 32
- 权重与偏置形状与输入兼容
"""
import torch
import torch.nn as nn
def mlp_forward(X:torch.Tensor,W1:torch.Tensor,b1:torch.Tensor,W2:torch.Tensor,b2:torch.Tensor)->torch.Tensor:
    X =torch.as_tensor(X,dtype=torch.float64)
    W1=torch.as_tensor(W1,dtype=torch.float64)
    b1=torch.as_tensor(b1,dtype=torch.float64)
    W2=torch.as_tensor(W2,dtype=torch.float64)
    b2=torch.as_tensor(b2,dtype=torch.float64)
    h = torch.relu(X@W1+b1)
    Y = h@W2+b2
    return Y.squeeze(-1) if Y.shape[-1] == 1 else Y


