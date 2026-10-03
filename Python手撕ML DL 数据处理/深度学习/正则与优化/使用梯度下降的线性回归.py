# 使用梯度下降的线性回归
# 难度：简单
# 领域：机器学习
#
# 编写一个使用梯度下降执行线性回归的 Python 函数。
# 该函数接受 NumPy 数组 X（特征矩阵，包含一列用于截距的全 1 列）和 y（目标向量）作为输入，
# 同时接受学习率 alpha 和迭代次数 iterations。
# 返回学到的系数（权重）作为一个 NumPy 数组。
#
# 要求：
# 最小化均方误差（MSE）损失函数：
# L(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
# 其中 h_\theta(x) = X\theta 是预测值，m 是样本数量。1/2 的系数简化了梯度计算。
#
# - 将所有权重初始化为零。
# - 使用批量梯度下降（在每次迭代中使用所有样本）。
# - 输入矩阵 X 的形状为 (m, n)，其中 m 是训练样本数，n 是特征数（包含代表偏置的全 1 列）。
# - 目标向量 y 的形状为 (m,)。
#
# 示例：
# 输入：
# X = np.array([[1, 1], [1, 2], [1, 3]]), y = np.array([3, 5, 7]), alpha = 0.1, iterations = 1000
#
# 输出：
# [1.0, 2.0]
#
# 推导步骤：
# 数据完美遵循 y = 1 + 2x。从 \theta = [0, 0] 开始，梯度下降迭代更新权重，
# 直到收敛到大约 [1, 2]，分别代表 1 的截距和 2 的斜率。

import torch

def linear_regression_gradient_descent(x:torch.Tensor,y:torch.Tensor,alpha:float=0.1,iterations:int=1000)->torch.Tensor:
    x = torch.as_tensor(x,dtype = torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    assert x.ndim == 2
    m,n = x.shape
    assert y.numel() == m
    assert alpha>0 and iterations>0
    #将权重theta 全部初始化为(n,) 的全0张量
    theta = torch.zeros(n,dtype=torch.float64)
    for _ in range(iterations):
        error = x @ theta - y
        grad = (x.T @ error) / m
        theta = theta - alpha * grad

    return theta


if __name__ == "__main__":
    # 示例测试: y = 1 + 2*x (目标截距为 1，斜率为 2)
    # 第一列全 1 代表截距特征，第二列代表自变量 x
    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0]
    ]
    y = [3.0, 5.0, 7.0]

    res = linear_regression_gradient_descent(X, y, alpha=0.1, iterations=1000)
    print("梯度下降收敛结果:", res)

    # 验证是否收敛到 [1.0, 2.0]
    expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert torch.allclose(res, expected, atol=1e-3), "测试未通过！"
    print("[PASS] 测试通过！模型成功拟合出最优权重：截距 1.0, 斜率 2.0！")


