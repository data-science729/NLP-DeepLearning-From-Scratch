import random
import numpy as np


def gradcheck_naive(f, x):
    """对目标函数 f 在点 x 处进行数值梯度检查。

    参数:
    f -- 目标函数，接受输入 x，返回一个元组 (cost, grad)，
         其中 cost 为标量损失，grad 为关于 x 的解析梯度。
    x -- 需要检查梯度的点（标量或任意维度的 NumPy 数组）。
    """
    # 保存当前的随机数生成器状态，确保每次计算 f(x) 时的随机性环境一致
    rndstate = random.getstate()
    random.setstate(rndstate)
    
    # 获取在原始输入 x 下的损失和解析梯度（反向传播计算所得）
    cost, grad = f(x)
    h = 1e-4  # 双边差分的微小扰动步长

    # 使用 np.nditer 遍历多维数组 x 的每一个元素
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index  # 当前元素的坐标元组（如一维的 (0,) 或二维的 (1, 2)）

        # 记录原始数值，准备施加微扰
        old_val = x[ix]

        # 1. 计算 f(x + h)
        x[ix] = old_val + h
        random.setstate(rndstate)
        cost_plus, _ = f(x)

        # 2. 计算 f(x - h)
        x[ix] = old_val - h
        random.setstate(rndstate)
        cost_minus, _ = f(x)

        # 3. 关键点：将 x[ix] 还原为原始值，避免污染后续元素的梯度计算
        x[ix] = old_val

        # 4. 根据双边（中心）差分公式计算数值梯度: (f(x+h) - f(x-h)) / (2*h)
        numgrad = (cost_plus - cost_minus) / (2.0 * h)

        # 5. 计算解析梯度与数值梯度之间的相对误差
        # 分母使用 max(1, abs(numgrad), abs(grad[ix])) 避免除以 0，并对大梯度进行尺度归一化
        reldiff = abs(numgrad - grad[ix]) / max(1.0, abs(numgrad), abs(grad[ix]))

        # 如果相对误差大于 1e-5，则判定梯度检查失败
        if reldiff > 1e-5:
            print("梯度检查失败（Gradient check failed!）")
            print("在索引 %s 处检测到首个梯度计算错误：" % str(ix))
            print("你计算的解析梯度: %f \t 真实的数值梯度: %f \t 相对误差: %e" % (grad[ix], numgrad, reldiff))
            return False

        it.iternext()

    print("梯度检查通过（Gradient check passed!）")
    return True


def sanity_check():
    """
    基础健全性检查。
    使用简单的二次函数 f(x) = sum(x^2) 进行验证，
    其损失为 sum(x^2)，理论解析梯度为 2 * x。
    """
    # 使用 np.atleast_1d 确保即使输入是 0 维标量数组，也能保持为 ndarray 进行求和，
    # 彻底避免 NumPy 2.x 在特定 IDE（如 PyCharm）环境下对纯标量求和触发的 _NoValueType 兼容性错误
    quad = lambda x: (np.sum(np.atleast_1d(x) ** 2), x * 2.0)

    print("正在运行基础健全性检查...")
    print("1. 标量输入测试:")
    assert gradcheck_naive(quad, np.array(123.456)), "标量测试未通过"

    print("2. 一维向量测试:")
    assert gradcheck_naive(quad, np.random.randn(3,)), "一维向量测试未通过"

    print("3. 二维矩阵测试:")
    assert gradcheck_naive(quad, np.random.randn(4, 5)), "二维矩阵测试未通过"
    print("基础健全性测试全部通过！\n")


def your_gradcheck_test():
    """
    自定义扩展测试（验证更复杂的非线性函数和极端情况）。
    直接运行：
        python q2_gradcheck.py
    该函数不会计入作业评分。
    """
    print("正在运行自定义测试...")
    # 自定义测试 1：三次多项式 f(x) = sum(x^3)，导数为 3 * x^2
    cubic = lambda x: (np.sum(np.atleast_1d(x) ** 3), 3.0 * (x ** 2))
    assert gradcheck_naive(cubic, np.random.randn(3, 4)), "三次函数梯度检查未通过"

    # 自定义测试 2：带有负数和偏置的复杂复合函数
    complex_fn = lambda x: (np.sum(np.sin(x) + np.cos(x)), np.cos(x) - np.sin(x))
    assert gradcheck_naive(complex_fn, np.random.randn(2, 3)), "复合三角函数梯度检查未通过"
    print("所有自定义测试均已通过！\n")


if __name__ == "__main__":
    sanity_check()
    your_gradcheck_test()
