"""
题目：动量优化器更新步（Momentum Optimizer）
难度：简单（Easy）
分类：深度学习（Deep Learning）

题目描述：
    实现动量优化器（Momentum Optimizer）的单步参数更新函数。
    该函数接收当前参数值（parameter）、梯度（grad）以及上一步累积的速度（velocity），
    计算并返回更新后的参数值与新速度。
    函数需同时支持标量（Scalar）以及多维数组/张量（Array/Tensor）形式的输入。

核心公式：
    1. 速度更新：v_{t+1} = β * v_t + lr * g_t
    2. 参数更新：θ_{t+1} = θ_t - v_{t+1}
    其中 β 为动量系数（momentum），lr 为学习率（learning rate）。

参数：
    parameter (torch.Tensor or float): 当前参数值，支持标量或任意形状张量。
    grad (torch.Tensor or float): 当前参数的梯度，形状与 parameter 一致。
    velocity (torch.Tensor or float): 上一步的速度（动量项），形状与 parameter 一致。
    lr (float, optional): 学习率，默认值为 0.01。
    momentum (float, optional): 动量衰减系数 β，默认值为 0.9。

返回：
    Tuple[torch.Tensor or float, torch.Tensor or float]:
        包含 (更新后的参数, 更新后的速度) 的元组，保持与输入相同的类型与形状结构。

示例：
    输入：
        parameter = 1.0, grad = 0.1, velocity = 0.1
        （默认参数：lr = 0.01, momentum = 0.9）
    输出：
        (0.909, 0.091)
    解释：
        新速度 v = 0.9 * 0.1 + 0.01 * 0.1 = 0.09 + 0.001 = 0.091；
        更新后参数 θ = 1.0 - 0.091 = 0.909。
"""
import torch
from typing import Optional


def momentum_step(
    parameter: torch.Tensor | float,
    grad: torch.Tensor | float,
    velocity: torch.Tensor | float,
    lr: float = 0.01,
    momentum: float = 0.9
) -> tuple[torch.Tensor | float, torch.Tensor | float]:

    assert lr>0
    assert 0.0<=momentum<1.0
    new_velocity = momentum*velocity+lr*grad
    new_parameter = parameter-new_velocity
    return new_parameter,new_velocity


if __name__ == '__main__':
    print('=' * 50)
    print('开始执行【动量优化器更新步】测试套件')
    print('=' * 50)

    # 测 1: 官方标量示例 (验证精度与类型保留)
    p1, g1, v1 = 1.0, 0.1, 0.1
    new_p1, new_v1 = momentum_step(p1, g1, v1)
    assert isinstance(new_p1, float) and isinstance(new_v1, float), 'Test 1 失败: 标量输入未返回 float'
    assert abs(new_p1 - 0.909) < 1e-5, f'Test 1 失败: 参数不符 {new_p1}'
    assert abs(new_v1 - 0.091) < 1e-5, f'Test 1 失败: 速度不符 {new_v1}'
    print(f'Test 1 通过 (官方标量用例): param={new_p1:.4f}, velocity={new_v1:.4f}')

    # 测 2: 2D 多维张量 (验证形状与逐元素并行运算)
    p2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    g2 = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    v2 = torch.zeros_like(p2)
    new_p2, new_v2 = momentum_step(p2, g2, v2, lr=0.1, momentum=0.9)
    expected_v2 = 0.9 * v2 + 0.1 * g2
    expected_p2 = p2 - expected_v2
    assert isinstance(new_p2, torch.Tensor) and isinstance(new_v2, torch.Tensor), 'Test 2 失败: 未返回 Tensor'
    assert new_p2.shape == p2.shape and new_v2.shape == v2.shape, 'Test 2 失败: 形状改变'
    assert torch.allclose(new_p2, expected_p2), 'Test 2 失败: 参数张量值不匹配'
    assert torch.allclose(new_v2, expected_v2), 'Test 2 失败: 速度张量值不匹配'
    print(f'Test 2 通过 (2D 张量并行计算与形状保留): shape={new_p2.shape}')

    # 测 3: 物理特性验证 (零梯度下的惯性滑行)
    p3, g3, v3 = 5.0, 0.0, 1.0
    new_p3, new_v3 = momentum_step(p3, g3, v3, lr=0.01, momentum=0.9)
    # 没有推力 (g=0)，全靠惯性 (v=0.9*1.0=0.9)
    assert abs(new_v3 - 0.9) < 1e-5, 'Test 3 失败: 惯性衰减计算错误'
    assert abs(new_p3 - 4.1) < 1e-5, 'Test 3 失败: 惯性滑行位移计算错误'
    print(f'Test 3 通过 (无梯度下的惯性滑行): param={new_p3:.4f}, velocity={new_v3:.4f}')

    # 测 4: 零动量退化验证 (momentum=0 时等价于经典 SGD)
    p4, g4, v4 = 2.0, 0.5, 999.0  # 无论旧速度多大，momentum=0 都会清除旧速度
    new_p4, new_v4 = momentum_step(p4, g4, v4, lr=0.1, momentum=0.0)
    expected_v4 = 0.1 * 0.5  # lr * g
    expected_p4 = 2.0 - expected_v4  # p - lr * g
    assert abs(new_v4 - expected_v4) < 1e-5 and abs(new_p4 - expected_p4) < 1e-5, 'Test 4 失败'
    print(f'Test 4 通过 (momentum=0 退化为经典 SGD): param={new_p4:.4f}, velocity={new_v4:.4f}')

    # 测 5: 非法超参数断言拦截
    try:
        momentum_step(1.0, 0.1, 0.1, lr=-0.01)
        raise RuntimeError('未能拦截负学习率')
    except AssertionError:
        pass

    try:
        momentum_step(1.0, 0.1, 0.1, momentum=1.0)
        raise RuntimeError('未能拦截大于等于 1 的 momentum')
    except AssertionError:
        pass
    print('Test 5 通过 (非法超参数拦截正常)')

    print('\n[PASS] 所有 5 个测试用例全部通过！')
