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