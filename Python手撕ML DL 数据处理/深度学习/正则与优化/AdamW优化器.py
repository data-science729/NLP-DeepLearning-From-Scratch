"""
题目：2105. AdamW 优化器单步更新（AdamW Optimization Step）
难度：中等（Medium）
分类：PyTorch / 深度学习（Deep Learning）/ 优化算法（Optimization）

题目描述：
    实现解耦权重衰减的 AdamW 优化器单步参数更新算法。
    在经典 Adam 中，若使用 L2 正则化（即在损失函数中增加 0.5 * λ * ||θ||²），
    正则化项的导数 λθ 会直接并入梯度 g_t 中，随后被二阶动量（自适应学习率分母）缩放，
    导致梯度较大的参数权重衰减被过度抑制、梯度较小的参数权重衰减被过度放大。

    AdamW（Loshchilov & Hutter, 2017）通过将权重衰减项（Weight Decay）与自适应梯度更新项解耦，
    直接在参数更新步上按固定比例缩小权重，恢复了真正的权重衰减效果。

核心公式：
    给定时间步 t >= 1，当前参数 θ、梯度 g、一阶动量 m、二阶动量 v、
    学习率 lr、动量衰减系数 β1 与 β2、数值稳定项 ε、权重衰减系数 λ：

    1. 更新有偏一阶矩与二阶矩：
       m_t = β1 * m + (1 - β1) * g
       v_t = β2 * v + (1 - β2) * (g²)
    2. 计算偏差修正（Bias Correction）：
       m̂_t = m_t / (1 - β1^t)
       v̂_t = v_t / (1 - β2^t)
    3. 参数解耦更新：
       θ_new = θ - lr * (m̂_t / (sqrt(v̂_t) + ε) + λ * θ)

参数：
    param (torch.Tensor or float): 当前参数值 θ，支持标量或多维张量。
    grad (torch.Tensor or float): 当前步的梯度 g，形状与 param 一致。
    m (torch.Tensor or float): 上一步累积的一阶动量矩，形状与 param 一致。
    v (torch.Tensor or float): 上一步累积的二阶动量矩，形状与 param 一致。
    t (int): 当前迭代时间步（t >= 1）。
    weight_decay (float): 权重衰减系数 λ（λ >= 0）。
    lr (float, optional): 学习率，默认值为 0.001。
    beta1 (float, optional): 一阶矩衰减系数，默认值为 0.9。
    beta2 (float, optional): 二阶矩衰减系数，默认值为 0.999。
    eps (float, optional): 防止除以零的数值稳定项，默认值为 1e-8。

返回：
    Tuple[Union[torch.Tensor, float], Union[torch.Tensor, float], Union[torch.Tensor, float]]:
        返回三元组 (新参数 θ_new, 新一阶矩 m_t, 新二阶矩 v_t)。

评测要求：
    与基准答案的绝对误差 <= 1e-5 或相对误差 <= 1e-5。

示例 1：
    输入：
        param = 1.0, grad = 0.1, m = 0.0, v = 0.0, t = 1, weight_decay = 0.01
        （默认参数：lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8）
    输出：
        (0.99899, 0.01, 0.00001)
    解释：
        1. 动量更新：
           m_1 = 0.9 * 0 + (1 - 0.9) * 0.1 = 0.01
           v_1 = 0.999 * 0 + (1 - 0.999) * (0.1²) = 0.001 * 0.01 = 0.00001
        2. 偏差修正：
           m̂_1 = 0.01 / (1 - 0.9¹) = 0.01 / 0.1 = 0.1
           v̂_1 = 0.00001 / (1 - 0.999¹) = 0.00001 / 0.001 = 0.01
        3. 自适应步长项：
           step_norm = m̂_1 / (sqrt(v̂_1) + eps) = 0.1 / (0.1 + 1e-8) ≈ 1.0
        4. 权重衰减与参数更新：
           decay_term = λ * θ = 0.01 * 1.0 = 0.01
           θ_new = 1.0 - 0.001 * (1.0 + 0.01) = 1.0 - 0.00101 = 0.99899
"""
import torch
from typing import Tuple, Union


def adamw_optimizer(
    param: torch.Tensor | float,
    grad: torch.Tensor | float,
    m: torch.Tensor | float,
    v: torch.Tensor | float,
    t: int,
    weight_decay: float = 0.01,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> tuple[torch.Tensor | float, torch.Tensor | float, torch.Tensor | float]:
    # 1. 基础校验
    assert t >= 1, "时间步 t 必须大于等于 1"
    assert lr > 0 and 0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0, "超参数取值不合法"
    assert eps > 0, "eps 必须大于 0"
    assert weight_decay >= 0.0, "weight_decay 必须大于等于 0"
    if isinstance(param, torch.Tensor):
        assert isinstance(grad, torch.Tensor) and isinstance(m, torch.Tensor) and isinstance(v, torch.Tensor), \
            "当 param 为 Tensor 时，grad, m, v 必须都为 Tensor"
        assert param.shape == grad.shape == m.shape == v.shape, "张量形状必须完全一致"

    # 2. 一阶矩与二阶矩更新 (纯梯度的指数移动平均 EMA)
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * (grad ** 2)

    # 3. 偏差修正 (分母加括号防运算优先级错误)
    m_hat = m_new / (1.0 - beta1 ** t)
    v_hat = v_new / (1.0 - beta2 ** t)

    # 4. 解耦权重衰减更新 (Decoupled Weight Decay)
    # 自适应步长项：m_hat / (sqrt(v_hat) + eps)
    # 独立解耦权重衰减项：weight_decay * param
    param_new = param - lr * (m_hat / ((v_hat ** 0.5) + eps) + weight_decay * param)

    return param_new, m_new, v_new


# 别名兼容（支持以 adamw_step 或 adamw_optimizer 调用）
adamw_step = adamw_optimizer


if __name__ == "__main__":
    print("=" * 50)
    print("开始执行【2105. AdamW 优化器单步更新】测试套件")
    print("=" * 50)

    # 测 1: 题目官方标准示例 1 (标量输入)
    p1 = 1.0
    g1 = 0.1
    m1 = 0.0
    v1 = 0.0
    t1 = 1
    wd1 = 0.01

    new_p1, new_m1, new_v1 = adamw_optimizer(p1, g1, m1, v1, t1, weight_decay=wd1)
    print(f"Test 1 输出: p={new_p1:.6f}, m={new_m1:.6f}, v={new_v1:.6f}")

    assert abs(new_p1 - 0.99899) < 1e-5, f"Test 1 参数更新错误: {new_p1}"
    assert abs(new_m1 - 0.01) < 1e-5, f"Test 1 一阶矩错误: {new_m1}"
    assert abs(new_v1 - 0.00001) < 1e-5, f"Test 1 二阶矩错误: {new_v1}"
    print("Test 1 通过 (官方标量示例精度完全吻合)")

    # 测 2: 2D 多维张量并行更新 + 非原地修改验证
    p2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    g2 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
    m2 = torch.zeros_like(p2)
    v2 = torch.zeros_like(p2)
    p2_orig = p2.clone()

    new_p2, new_m2, new_v2 = adamw_optimizer(p2, g2, m2, v2, t=1, weight_decay=0.01)

    assert new_p2.shape == p2.shape and new_m2.shape == m2.shape and new_v2.shape == v2.shape, "Test 2 形状不匹配"
    assert new_p2 is not p2 and new_m2 is not m2 and new_v2 is not v2, "安全警告: 未返回新张量！"
    assert torch.allclose(p2, p2_orig), "安全警告: 输入原始参数被篡改！"
    print(f"Test 2 通过 (2D 张量并行计算与非原地修改验证): shape={new_p2.shape}")

    # 测 3: weight_decay=0 时退化为标准 Adam
    p3, g3, m3, v3 = 1.0, 0.1, 0.0, 0.0
    out_adamw_zero_wd, _, _ = adamw_optimizer(p3, g3, m3, v3, t=1, weight_decay=0.0)
    # 当 weight_decay=0 时，结果应精确等于前一题 Adam 的 0.9990000001
    assert abs(out_adamw_zero_wd - 0.9990000001) < 1e-5, "Test 3 失败: weight_decay=0 时未能退化为 Adam"
    print("Test 3 通过 (weight_decay=0 精确退化为经典 Adam)")

    # 测 4: 异常断言拦截 (t=0 或非法超参数)
    try:
        adamw_optimizer(1.0, 0.1, 0.0, 0.0, t=0)
        raise RuntimeError("未能拦截 t=0 非法输入")
    except AssertionError:
        pass
    print("Test 4 通过 (非法输入断言拦截正常)")

    print("\n[PASS] 所有 4 个测试用例全部完美通过！")
