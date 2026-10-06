"""
题目：带有线性预热的余弦衰减学习率调度（Warmup + Cosine Decay Schedule）
难度：中等（Medium）
分类：优化算法（Optimization）/ 深度学习

题目描述：
    实现一个结合线性预热（Linear Warmup）与余弦衰减（Cosine Decay）的学习率调度算法。

    调度逻辑分为两个阶段：
    1. 预热阶段（Warmup Phase）：
       在训练的前 W 个步长（即 step 0 到 step W - 1）内，学习率从 0.0 线性递增至最大学习率 lr_max。
    2. 余弦衰减阶段（Cosine Decay Phase）：
       预热结束后，从 step W 开始，学习率以 lr_max 为起点遵循余弦退火曲线平滑衰减，
       并在总步数 T 处达到设定的最小学习率 lr_min。

    函数需计算并返回一个长度为 T 的浮点数列表，表示整个训练周期（step 0 到 T - 1）中
    每个步长对应的学习率。数值通常四舍五入保留 4 位小数。

核心公式：
    - 阶段一：线性预热 (0 <= t < W)
      lr_t = lr_max * (t / W)

    - 阶段二：余弦衰减 (W <= t < T)
      衰减区间步长数 progress = (t - W) / (T - W)
      lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(progress * π))

参数：
    T (int): 总训练步数（Total training steps）。
    W (int): 线性预热步数（Warmup steps）。
    lr_max (float): 最大学习率（预热阶段的目标峰值）。
    lr_min (float): 最小学习率（余弦衰减的目标下限）。

返回：
    List[float]: 长度为 T 的列表，包含从 step 0 到 step T - 1 的所有学习率。

示例：
    输入：
        T = 10, W = 3, lr_max = 1.0, lr_min = 0.0
    输出：
        [0.0, 0.3333, 0.6667, 1.0, 0.9505, 0.8117, 0.6113, 0.3887, 0.1883, 0.0495]
    解释：
        - 预热阶段（W = 3 步）：
          step 0: 1.0 * (0 / 3) = 0.0
          step 1: 1.0 * (1 / 3) ≈ 0.3333
          step 2: 1.0 * (2 / 3) ≈ 0.6667
        - 余弦衰减阶段（剩余 7 步，T - W = 7）：
          step 3: (3 - 3) / 7 = 0/7 -> cos(0) = 1.0 -> lr = 1.0
          step 4: (4 - 3) / 7 = 1/7 -> cos(π/7) ≈ 0.9010 -> lr ≈ 0.9505
          step 5: (5 - 3) / 7 = 2/7 -> cos(2π/7) ≈ 0.6235 -> lr ≈ 0.8117
          step 6: (6 - 3) / 7 = 3/7 -> cos(3π/7) ≈ 0.2225 -> lr ≈ 0.6113
          step 7: (7 - 3) / 7 = 4/7 -> cos(4π/7) ≈ -0.2225 -> lr ≈ 0.3887
          step 8: (8 - 3) / 7 = 5/7 -> cos(5π/7) ≈ -0.6235 -> lr ≈ 0.1883
          step 9: (9 - 3) / 7 = 6/7 -> cos(6π/7) ≈ -0.9010 -> lr ≈ 0.0495
"""
import torch


def warmup_cosine_schedule(T: int, W: int, lr_max: float, lr_min: float) -> torch.Tensor:
    # 1. 基础断言
    assert T > 0, "总步数 T 必须大于 0"
    assert 0 <= W <= T, "预热步数 W 必须在 [0, T] 之间"
    assert lr_max >= lr_min >= 0, "学习率需满足 lr_max >= lr_min >= 0"

    # 2. 创建步长张量与结果张量 (float64 保证计算精度)
    t = torch.arange(T, dtype=torch.float64)
    lrs = torch.empty(T, dtype=torch.float64)

    # 3. 向量化预热阶段 (前 W 步)
    if W > 0:
        lrs[:W] = lr_max * (t[:W] / W)

    # 4. 向量化余弦衰减阶段 (后 T - W 步)
    if W < T:
        progress = (t[W:] - W) / (T - W)
        lrs[W:] = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + torch.cos(progress * torch.pi))

    # 5. 张量四舍五入保留 4 位小数
    return torch.round(lrs, decimals=4)






