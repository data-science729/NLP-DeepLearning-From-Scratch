# 1406. Adam 优化步
# 中等
# 标签：深度学习 / 优化算法
#
# Adam（Adaptive Moment Estimation）是一种常用的梯度优化算法，
# 通过维护梯度的一阶矩和二阶矩估计，对模型参数进行自适应更新。
# 参考文献：Adam: A Method for Stochastic Optimization
#
# 给定当前参数 theta、当前梯度 g、以及上一时刻的一阶矩 mprev 和二阶矩 vprev，
# 实现一次 Adam 更新步骤。
#
# 实现要求：
# 默认 lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8
# 返回 (新参数, m, v)
#
# 算法公式：
# m_t = beta1 * m_{t-1} + (1 - beta1) * g
# v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
# \hat{m} = m_t / (1 - beta1^t)
# \hat{v} = v_t / (1 - beta2^t)
# \theta \leftarrow \theta - lr * \hat{m} / (\sqrt{\hat{v}} + \epsilon)
#
# 输出与判定：
# 返回三元组；浮点比较：绝对误差 <= 10^-5 或相对误差 <= 10^-5（满足其一即通过）。
#
# 示例 1：
# 输入：
# parameter = 1, grad = 0.1, m = 0, v = 0, t = 1
# 输出：
# (0.9990000001, 0.009999999999999998, 1.0000000000000011e-05)
#
# 限制条件：
# 参数为标量测例
# t >= 1
def adam_optimizer(
    parameter: float,
    grad: float,
    m: float,
    v: float,
    t: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> tuple[float, float, float]:
    # 1. 基础断言校验
    assert t >= 1, "时间步 t 必须大于等于 1（用于偏差修正）"
    assert lr > 0 and 0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0 and eps > 0, "超参数取值不合法"
    m_new = beta1*m + (1-beta1)*grad
    v_new = beta2*v +(1-beta2)*(grad**2)
    m_hat = m_new/(1.0-beta1**t)
    v_hat = v_new/(1.0-beta2**t)
    new_parameter = parameter - (lr*m_hat)/((v_hat**0.5)+eps)
    # 返回题目要求的三元组：(新参数, 新的一阶矩, 新的二阶矩)
    return new_parameter, m_new, v_new
if __name__ == "__main__":
    print("=" * 50)
    print("开始执行【1406. Adam 优化步】测试套件")
    print("=" * 50)

    # 测 1: 题目官方标准示例 1
    p1, g1, m1, v1, t1 = 1.0, 0.1, 0.0, 0.0, 1
    new_p, new_m, new_v = adam_optimizer(p1, g1, m1, v1, t1)

    print(f"输出结果: ({new_p}, {new_m}, {new_v})")
    assert abs(new_p - 0.9990000001) < 1e-5, f"参数更新错误: {new_p}"
    assert abs(new_m - 0.009999999999999998) < 1e-5, f"一阶矩错误: {new_m}"
    assert abs(new_v - 1.0000000000000011e-05) < 1e-5, f"二阶矩错误: {new_v}"
    print("Test 1 通过 (官方标准示例精度完全匹配)")

    # 测 2: 验证第 2 步迭代 (t=2)，保证状态能够连续接力
    new_p2, new_m2, new_v2 = adam_optimizer(new_p, g1, new_m, new_v, t=2)
    assert new_p2 < new_p, "连续正向梯度下，参数应持续单调减小"
    print(f"Test 2 通过 (多步迭代连续更新正常): p2={new_p2:.6f}")

    # 测 3: 异常边界校验 (拦截 t < 1)
    try:
        adam_optimizer(1.0, 0.1, 0.0, 0.0, t=0)
        raise RuntimeError("未能拦截 t=0 非法输入")
    except AssertionError:
        pass
    print("Test 3 通过 (非法 t=0 断言拦截成功)")

    print("\n[PASS] 所有测试完美通过！")

