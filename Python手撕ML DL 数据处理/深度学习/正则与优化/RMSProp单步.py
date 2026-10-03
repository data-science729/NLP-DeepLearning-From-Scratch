"""
题目：RMSProp 优化器单步更新（Implement RMSProp Optimizer）
难度：中等（Medium）
分类：优化算法（Optimization）/ 深度学习（Deep Learning）

题目描述：
    实现均方根传递优化器（RMSProp Optimizer）的单步参数更新逻辑。
    RMSProp 通过维护历史梯度平方的指数加权移动平均（Exponential Moving Average, EMA）
    来为各个参数独立自适应调整学习率：
    - 对于梯度持续较大的参数方向，累积均方根较大，从而缩小其实际更新步长以抑制震荡；
    - 对于梯度较小或稀疏的参数方向，累积均方根较小，从而放大其实际更新步长以加速收敛。

    给定当前参数值 params、当前梯度 grads、历史梯度平方缓存 cache、学习率 lr、
    衰减系数 beta 以及数值稳定项 eps（默认 1e-8）：
    计算并返回更新后的参数值与更新后的缓存。

核心公式：
    1. 梯度平方移动平均更新（Cache 更新）：
       v_{new} = beta * v + (1 - beta) * g²
    2. 参数更新（注意 eps 在根号外）：
       p_{new} = p - lr * (g / (sqrt(v_{new}) + eps))

参数：
    params (List[float]): 当前参数数值列表。
    grads (List[float]): 当前参数对应的梯度数值列表，与 params 形状一致。
    cache (List[float]): 历史梯度平方的指数移动平均缓存列表，与 params 形状一致。
    lr (float): 学习率（Learning Rate）。
    beta (float): 衰减率 / 平滑常数（Decay Rate），通常取 0.9。
    eps (float, optional): 防止分母为零的数值稳定常数，默认为 1e-8。

返回：
    Tuple[List[float], List[float]]:
        包含 (更新后的参数列表 params_new, 更新后的缓存列表 cache_new) 的二元组。

示例：
    输入：
        params = [1.0], grads = [0.1], cache = [0.0], lr = 0.1, beta = 0.9
    输出：
        ([0.683772], [0.001])
    解释：
        1. 更新缓存 v_{new}：
           v_{new} = 0.9 * 0.0 + (1 - 0.9) * (0.1²)
                   = 0.0 + 0.1 * 0.01
                   = 0.001
        2. 计算分母有效步长：
           sqrt(v_{new}) + eps = sqrt(0.001) + 1e-8
                               ≈ 0.0316227766 + 1e-8
                               ≈ 0.0316227866
        3. 计算参数更新量：
           delta = 0.1 * (0.1 / 0.0316227866)
                 = 0.01 / 0.0316227866
                 ≈ 0.316227766
        4. 更新参数 p_{new}：
           p_{new} = 1.0 - 0.316227766 ≈ 0.683772
"""