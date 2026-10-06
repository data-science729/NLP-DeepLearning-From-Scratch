"""
问题描述：
实现带有均方误差（MSE）损失函数的梯度下降变体
难度：中等
分类：机器学习

题目概述：
在本题中，你需要实现一个能够执行三种梯度下降变体的单一函数：
随机梯度下降（SGD）、批量梯度下降（Batch GD）和小批量梯度下降（Mini-Batch GD），
并使用均方误差（MSE）作为损失函数。该函数需要接受一个额外的参数来指定所使用的变体。

要求：
- 不要对数据进行混洗（shuffling）；必须按原始顺序处理样本（索引 0, 1, 2, ...）
- 批量梯度下降（Batch GD）：每个 epoch 使用所有样本计算一次梯度更新
- 随机梯度下降（Stochastic GD）：按顺序逐个遍历每个样本（即先处理样本 0，然后是 1、2 等），而不是随机选择
- 小批量梯度下降（Mini-Batch GD）：由连续样本构成不重叠的批次（例如，当 batch_size=2 时：第一个批次使用索引 [0, 1]，第二个批次使用 [2, 3]，以此类推）
- `n_epochs` 参数指定对数据集进行完整遍历的次数
- 在每个 epoch 中，按照指定的方法处理所有样本

示例：
输入：
import numpy as np

# 样本数据
X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
y = np.array([2, 3, 4, 5])

# 参数
learning_rate = 0.01
n_epochs = 1000
batch_size = 2

# 初始化权重
weights = np.zeros(X.shape[1])

# 测试批量梯度下降
final_weights = gradient_descent(X, y, weights, learning_rate, n_epochs, method='batch')
# 测试随机梯度下降
final_weights = gradient_descent(X, y, weights, learning_rate, n_epochs, method='stochastic')
# 测试小批量梯度下降
final_weights = gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size, method='mini_batch')

输出：
[float, float]
[float, float]
[float, float]

推理说明：
该函数应当针对给定的 epoch 数量，在完成对数据的指定梯度下降变体完整遍历后，返回最终的权重。
"""
import torch


def gradient_descent_variants(x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor,
                     alpha:float=0.01, epochs: int=1000,
                     batch_size: int = 2, method: str = 'batch') -> torch.Tensor:
    x = torch.as_tensor(x,dtype = torch.float64)
    y= torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    theta = torch.as_tensor(theta,dtype = torch.float64)
    assert x.ndim == 2
    m,n = x.shape
    assert y.numel() == m
    assert alpha>0 and epochs>0
    assert 1<=batch_size<=m
    assert method in ["batch","stochastic","mini_batch"]
    theta = torch.zeros(n,dtype = torch.float64)
    for _ in range(epochs):
        for start_idx in range(0,m,batch_size):
            end_idx = min(start_idx + batch_size,m)
            x_batch  =x[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            curr_b = end_idx-start_idx  #当前批的实际样本数(防最后一个batch不足)
            #算残差和梯度
            error = x_batch@theta - y_batch
            grad = (x_batch.T@error)/curr_b
            #参数更新
            theta = theta-alpha*grad
    return theta
if __name__ == "__main__":
    # y = 1 + 2*x
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
    y = [3.0, 5.0, 7.0, 9.0]
    m = len(y)

    # 1. 测试 BGD 模式 (batch_size = m = 4)
    w_bgd = gradient_descent_variants(X, y, batch_size=m, alpha=0.1, epochs=500)
    print("BGD 收敛结果:", w_bgd)

    # 2. 测试 SGD 模式 (batch_size = 1)
    w_sgd = gradient_descent_variants(X, y, batch_size=1, alpha=0.01, epochs=500)
    print("SGD 收敛结果:", w_sgd)

    # 3. 测试 Mini-batch 模式 (batch_size = 2)
    w_mb = gradient_descent_variants(X, y, batch_size=2, alpha=0.05, epochs=500)
    print("Mini-batch 收敛结果:", w_mb)







