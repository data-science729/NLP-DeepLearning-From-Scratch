import numpy as np


def sigmoid(x):
    """计算输入 x 的 Sigmoid 激活函数值。

    参数:
    x -- 标量或任意维度的 NumPy 数组。

    返回值:
    s -- sigmoid(x)，形状与输入 x 相同。
    """
    # 按照 Sigmoid 公式计算: 1 / (1 + e^(-x))
    s = 1.0 / (1.0 + np.exp(-x))
    return s


def sigmoid_grad(s):
    """计算 Sigmoid 函数关于输入 x 的导数（梯度）。

    注意：根据 2(a) 题推导出的性质 σ'(x) = σ(x) * (1 - σ(x))，
    该函数的入参 s 是已经计算好的 Sigmoid 输出（即 s = sigmoid(x)），
    而不是原始的输入 x。

    参数:
    s -- Sigmoid 函数的输出值（标量或任意维度的 NumPy 数组）。

    返回值:
    ds -- Sigmoid 关于输入 x 的梯度，形状与 s 相同。
    """
    # 利用公式 σ'(x) = s * (1 - s) 计算梯度
    ds = s * (1.0 - s)
    return ds


def test_sigmoid_basic():
    """
    基础测试用例。
    注意：这些测试并不全面。
    """
    print("正在运行 Sigmoid 基础测试...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print("f(x) 输出:\n", f)
    print("g(s) 输出:\n", g)

    # 验证 Sigmoid 前向计算精度
    f_ans = np.array([[0.73105858, 0.88079708], [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06), "Sigmoid 前向计算结果不符"

    # 验证 Sigmoid 梯度计算精度
    g_ans = np.array([[0.19661193, 0.10499359], [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06), "Sigmoid 梯度计算结果不符"

    print("基础测试通过！\n")


def test_sigmoid():
    """
    可以在此函数中编写自定义测试，直接运行：
        python q2_sigmoid.py
    该测试函数不会计入作业评分。
    """
    print("正在运行自定义测试...")
    # 1. 测试标量输入
    assert np.isclose(sigmoid(0.0), 0.5), "sigmoid(0) 应该等于 0.5"
    assert np.isclose(sigmoid_grad(sigmoid(0.0)), 0.25), "sigmoid_grad 在 0 处应该等于 0.25"

    # 2. 测试典型正负值边界
    boundary_x = np.array([10.0, -10.0])
    s_boundary = sigmoid(boundary_x)
    assert np.isclose(s_boundary[0], 1.0, atol=1e-4), "较大正数的 Sigmoid 应该接近 1"
    assert np.isclose(s_boundary[1], 0.0, atol=1e-4), "较小负数的 Sigmoid 应该接近 0"
    grad_boundary = sigmoid_grad(s_boundary)
    assert np.all(grad_boundary >= 0.0), "Sigmoid 导数必须非负"
    print("所有自定义测试均已通过！\n")


if __name__ == "__main__":
    test_sigmoid_basic()
    test_sigmoid()
