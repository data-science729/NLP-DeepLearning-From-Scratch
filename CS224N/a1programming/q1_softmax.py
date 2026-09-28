import numpy as np


def softmax(x):
    """计算输入 x 每一行的 softmax 函数值。

    注意：此函数必须是完全向量化实现的，即使用 NumPy 矩阵/数组操作，
    而不是使用 Python 的 for 循环。

    参数:
    x -- N 维向量，或者 M x N 维的 NumPy 矩阵。

    返回值:
    x -- 形状与输入相同，允许在原数组上直接修改。
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # 矩阵情况：对每一行 (axis=1) 分别计算 softmax
        # 1. 减去每行最大值 c = -max(x)，保证数值稳定性（防止指数过大导致上溢 Overflow）
        # 2. keepdims=True 保持形状为 (M, 1)，利用 NumPy 广播机制进行计算，避免使用 for 循环
        max_x = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_x)
        x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        # 向量情况：输入为 1D 向量
        # 减去向量中的最大值保证数值稳定性
        max_x = np.max(x)
        exp_x = np.exp(x - max_x)
        x = exp_x / np.sum(exp_x)

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    基础测试用例。
    注意：这些测试并不全面。
    """
    print("正在运行基础测试...")
    test1 = softmax(np.array([1, 2]))
    print("test1 输出:\n", test1)
    ans1 = np.array([0.26894142, 0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print("test2 输出:\n", test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print("test3 输出:\n", test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("你应该可以通过手算验证这些结果！\n")


def test_softmax():
    """
    可以在此函数中编写自定义测试，直接运行：
        python q1_softmax.py
    该测试函数不会计入作业评分。
    """
    print("正在运行自定义测试...")
    # 测试多维矩阵（随机矩阵每一行的概率和应为 1）
    rng = np.random.RandomState(42)
    test_mat = rng.randn(10, 5)
    out_mat = softmax(test_mat)
    # 验证每一行的和是否接近 1
    row_sums = np.sum(out_mat, axis=1)
    assert np.allclose(row_sums, np.ones(10)), "每一行的和都应该为 1"
    
    # 测试极端大/小数值下的数值稳定性（防止产生 NaN 或 Inf）
    extreme_test = softmax(np.array([[1e6, 1e6 + 1], [-1e6, -1e6 - 1]]))
    assert not np.isnan(extreme_test).any(), "极端数值测试中出现了 NaN"
    assert not np.isinf(extreme_test).any(), "极端数值测试中出现了 Inf"
    print("所有自定义测试均已通过！\n")


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()