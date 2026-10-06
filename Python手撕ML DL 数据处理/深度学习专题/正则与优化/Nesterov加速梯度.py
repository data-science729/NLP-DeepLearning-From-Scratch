"""
题目：Nesterov 加速梯度优化器更新步（Nesterov Accelerated Gradient Optimizer）
难度：简单（Easy）
分类：深度学习（Deep Learning）/ 优化算法（Optimization）

题目描述：
    使用经典的前瞻位置形式（Look-ahead Formulation，Sutskever 等人于 2013 年提出），
    实现 Nesterov 加速梯度（NAG）优化器的单步参数更新函数。

    该形式的核心思想是：在当前动量方向上预先跨出一步（前瞻位置），
    并在该位置计算梯度，利用“预知”的梯度来修正当前的累积速度与参数更新方向。

核心公式：
    1. 计算前瞻位置（Look-ahead position）：
       θ_lookahead = θ - γ * v
    2. 计算前瞻位置处的梯度：
       g = ∇J(θ_lookahead)
    3. 累积更新速度（Velocity update）：
       v_new = γ * v + η * g
    4. 更新模型参数（Parameter update）：
       θ_new = θ - v_new

    其中：
    - γ (gamma): 动量衰减系数（momentum），默认取 0.9；
    - η (eta): 学习率（learning rate / lr），默认取 0.01；
    - ∇J: 目标函数的梯度函数（grad_fn）。

参数：
    parameter (torch.Tensor or float): 当前参数值 θ，支持标量或任意形状张量。
    grad_fn (Callable): 梯度计算函数，输入为前瞻位置张量/标量，返回对应的梯度。
    velocity (torch.Tensor or float): 上一步累积的速度 v，形状与 parameter 一致。
    lr (float, optional): 学习率 η，默认值为 0.01。
    momentum (float, optional): 动量系数 γ，默认值为 0.9。

返回：
    Tuple[torch.Tensor or float, torch.Tensor or float]:
        包含 (更新后的参数 θ_new, 更新后的速度 v_new) 的元组，保持与输入相同的类型与维度。

示例：
    输入：
        parameter = 1.0, grad_fn = lambda x: x, velocity = 0.1
        （默认参数：lr = 0.01, momentum = 0.9）
    输出：
        (0.9009, 0.0991)
    解释：
        1. 前瞻位置：
           θ_lookahead = 1.0 - 0.9 * 0.1 = 1.0 - 0.09 = 0.91
        2. 前瞻梯度：
           g = grad_fn(0.91) = 0.91
        3. 速度更新：
           v_new = 0.9 * 0.1 + 0.01 * 0.91 = 0.09 + 0.0091 = 0.0991
        4. 参数更新：
           θ_new = 1.0 - 0.0991 = 0.9009
"""

import torch
from typing import Callable, Tuple, Union

def nesterov_momentum_step(
        parameter:torch.Tensor | float,
        grad_fn:Callable,
        velocity:torch.Tensor | float,
        lr:float = 0.01,
        momentum:float = 0.9
)->tuple[torch.Tensor | float,torch.Tensor | float]:
    #1.基础检验
    assert lr>0
    assert 0.0<=momentum<1.0
    assert callable(grad_fn)
    if isinstance(parameter,torch.Tensor) and isinstance(velocity,torch.Tensor):
        assert parameter.shape == velocity.shape
    #2.第一跳：借着惯性先跨出半步 得到前瞻位置
    theta_lookahead = parameter - momentum*velocity
    g = grad_fn(theta_lookahead)
    v_new = momentum*velocity + lr*g
    theta_new = parameter - v_new
    return theta_new,v_new


if __name__ == "__main__":
    print("=" * 50)
    print("开始执行【Nesterov 加速梯度优化器】测试套件")
    print("=" * 50)

    # 测 1: 官方标准标量示例
    p1 = 1.0
    grad_fn1 = lambda x: x
    v1 = 0.1
    lr1 = 0.01
    mom1 = 0.9

    new_p1, new_v1 = nesterov_momentum_step(p1, grad_fn1, v1, lr1, mom1)

    # 理论值：v = 0.0991, p = 0.9009
    assert abs(new_p1 - 0.9009) < 1e-4, f"Test 1 参数错误: {new_p1}"
    assert abs(new_v1 - 0.0991) < 1e-4, f"Test 1 速度错误: {new_v1}"
    print(f"Test 1 通过 (官方标量示例): param={new_p1:.4f}, velocity={new_v1:.4f}")

    # 测 2: 2D 多维张量测试 (例如二次凸碗损失 f(x) = x^2, grad_fn(x) = 2x)
    p2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    v2 = torch.zeros_like(p2)
    grad_fn2 = lambda x: 2.0 * x  # 梯度函数支持张量并行

    new_p2, new_v2 = nesterov_momentum_step(p2, grad_fn2, v2, lr=0.1, momentum=0.9)
    # v=0 时，lookahead = p2，g = 2*p2，v_new = 0.1 * 2 * p2 = 0.2 * p2
    expected_v2 = 0.2 * p2
    expected_p2 = p2 - expected_v2
    assert torch.allclose(new_p2, expected_p2), "Test 2 张量权重计算错误"
    assert torch.allclose(new_v2, expected_v2), "Test 2 张量速度计算错误"
    assert new_p2.shape == p2.shape, "Test 2 形状未保持"
    print(f"Test 2 通过 (2D 张量并行梯度计算与形状保留): shape={new_p2.shape}")

    # 测 3: grad_fn 非法输入拦截
    try:
        nesterov_momentum_step(p1, "not_a_function", v1)
        raise RuntimeError("未能拦截非函数 grad_fn")
    except AssertionError:
        pass
    print("Test 3 通过 (非法 grad_fn 断言拦截正常)")

    print("\n[PASS] 所有测试用例全部完美通过！")





