"""
2106. ResNet BasicBlock（PyTorch） (简单)

题目描述：
    实现一个由两层线性变换配合 ReLU 激活函数构成的基础残差块（BasicBlock）。
    子网络的主路径输出与输入恒等映射（Identity / Shortcut）直接相加后，
    再通过一次 ReLU 激活函数得到最终输出。

数学公式：
    y = ReLU(x + W2 @ ReLU(W1 @ x))

实现要求：
    - W1, W2 均为方阵，维度与输入向量 x 的维度严格一致。
    - 主路径为：Linear(W1) -> ReLU -> Linear(W2)。
    - 残差连接：将主路径输出与原始输入 x 逐元素相加后，再执行一次 ReLU。

参数说明：
    x (list[float] | torch.Tensor): 输入向量，维度为 d
    w1 (list[list[float]] | torch.Tensor): 第一层线性变换权重矩阵，形状为 (d, d)
    w2 (list[list[float]] | torch.Tensor): 第二层线性变换权重矩阵，形状为 (d, d)

返回值：
    list[float] | torch.Tensor: 经过残差块处理后的输出向量，维度与输入 x 一致

评测判定：
    - 返回与输入同长的向量。
    - 浮点比较满足绝对误差 <= 1e-5 或相对误差 <= 1e-5 即视为通过。

示例 1:
    输入:
        x = [1, 2]
        w1 (2, 2) = [[1, 0], [0, 1]]
        w2 (2, 2) = [[0.5, 0], [0, 0.5]]
    输出:
        返回值 = [1.5, 3.0]

限制条件:
    向量维数 d <= 32
"""
import torch
import torch.nn as nn
def residual_block(x:torch.Tensor,W1:torch.Tensor,W2:torch.Tensor)->torch.Tensor:
    x = torch.as_tensor(x,dtype=torch.float32)
    W1 = torch.as_tensor(W1,dtype=torch.float32)
    W2 =torch.as_tensor(W2,dtype = torch.float32)
    #第一层 矩阵乘法W1@x 随后经过RELU激活
    h1 = torch.relu(W1@x)
    #第二层 矩阵乘法W1@h1 产出主路径变换F(x)
    h2 = W2@h1
    #残差相加:原始输入x直接越过两层与h2相加 最后通过Relu
    h3 = torch.relu(x+h2)
    return h3
if __name__ == '__main__':
    print("=" * 60)
    print("测试用例 1: 题目示例 1")
    x1 = [1, 2]
    w1 = [[1, 0], [0, 1]]
    w2 = [[0.5, 0], [0, 0.5]]
    res1 = residual_block(x1, w1, w2)
    print(f"输入: x = {x1}")
    print(f"输出: {res1.tolist()}")
    print("预期输出: [1.5, 3.0]")
    assert torch.allclose(res1, torch.tensor([1.5, 3.0]), atol=1e-5), "用例 1 失败！"
    print(">>> 判定: 通过！<<<")

    print("\n" + "=" * 60)
    print("测试用例 2: 负数激活截断测试")
    x2 = [-2, 3]
    res2 = residual_block(x2, w1, w1)
    # W1@x = [-2, 3] -> ReLU -> [0, 3] -> W2@[0, 3] -> [0, 3]
    # x + h2 = [-2, 3] + [0, 3] = [-2, 6] -> ReLU -> [0, 6]
    print(f"输入: x = {x2}")
    print(f"输出: {res2.tolist()}")
    print("预期输出: [0.0, 6.0]")
    assert torch.allclose(res2, torch.tensor([0.0, 6.0]), atol=1e-5), "用例 2 失败！"
    print(">>> 判定: 通过！<<<")
    print("=" * 60)
