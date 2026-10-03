"""
题目：RMSProp 优化器单步更新（Implement RMSProp Optimizer）
难度：中等（Medium）
分类：PyTorch / 深度学习（Deep Learning）

题目描述：
    使用 PyTorch 实现均方根传递优化器（RMSProp Optimizer）的单步参数更新逻辑。
    RMSProp 通过维护历史梯度平方的指数加权移动平均（Exponential Moving Average, EMA）
    来为各个参数独立自适应调整学习率：
    - 对于梯度持续较大的参数方向，累积均方根较大，从而缩小其实际更新步长以抑制震荡；
    - 对于梯度较小或稀疏的参数方向，累积均方根较小，从而放大其实际更新步长以加速收敛。

    给定当前参数张量 params、梯度张量 grads、历史梯度平方缓存张量 cache、学习率 lr、
    衰减系数 beta 以及数值稳定常数 eps（默认 1e-8）：
    计算并以全新张量的形式返回更新后的参数值与更新后的缓存（禁止原地修改）。

核心公式：
    1. 梯度平方移动平均更新（Cache 更新）：
       v_{new} = beta * v + (1 - beta) * g²
    2. 参数更新（注意 eps 位于分母根号外）：
       p_{new} = p - lr * (g / (sqrt(v_{new}) + eps))

参数：
    params (torch.Tensor): 当前网络参数张量，支持任意维度。
    grads (torch.Tensor): 当前参数对应的梯度张量，形状与 params 一致。
    cache (torch.Tensor): 历史梯度平方的指数移动平均缓存张量，形状与 params 一致。
    lr (float): 学习率（Learning Rate）。
    beta (float): 平滑系数 / 衰减率（Decay Rate），通常取 0.9。
    eps (float, optional): 防止除以零的数值稳定常数，默认为 1e-8。

返回：
    Tuple[torch.Tensor, torch.Tensor]:
        包含 (更新后的参数张量 params_new, 更新后的缓存张量 cache_new) 的二元组。

示例：
    输入：
        params = torch.tensor([1.0])
        grads = torch.tensor([0.1])
        cache = torch.tensor([0.0])
        lr = 0.1
        beta = 0.9
        eps = 1e-8
    输出：
        (tensor([0.6838]), tensor([0.0010]))
    解释：
        1. 更新缓存 v_{new}：
           v_{new} = 0.9 * 0.0 + (1 - 0.9) * (0.1²)
                   = 0.001
        2. 计算分母有效步长：
           sqrt(v_{new}) + eps = sqrt(0.001) + 1e-8 ≈ 0.0316227866
        3. 计算参数更新量并更新：
           p_{new} = 1.0 - 0.1 * (0.1 / 0.0316227866) ≈ 0.683772
"""
import torch
def RMSProp_step(
    params: torch.Tensor,
    grads: torch.Tensor,
    cache: torch.Tensor,
    lr: float,
    beta: float,
    eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    assert params.shape == grads.shape == cache.shape, "张量形状必须一致"
    assert lr > 0 and 0.0 <= beta < 1.0 and eps > 0, "超参数取值不合法"
    v_new = beta * cache + (1.0 - beta) * (grads ** 2)
    eff_lr = lr / (torch.sqrt(v_new) + eps)
    p_new = params - eff_lr * grads
    return p_new, v_new


if __name__ == "__main__":
    print("=" * 50)
    print("开始执行【RMSProp 优化器单步更新】测试套件")
    print("=" * 50)

    # 测 1: 官方标准示例
    p1 = torch.tensor([1.0], dtype=torch.float64)
    g1 = torch.tensor([0.1], dtype=torch.float64)
    c1 = torch.tensor([0.0], dtype=torch.float64)
    lr1, beta1, eps1 = 0.1, 0.9, 1e-8

    p1_orig = p1.clone()
    c1_orig = c1.clone()

    new_p1, new_c1 = RMSProp_step(p1, g1, c1, lr1, beta1, eps1)

    # 理论值：v = 0.001, p ≈ 0.683772
    expected_p1 = torch.tensor([0.68377223], dtype=torch.float64)
    expected_c1 = torch.tensor([0.0010], dtype=torch.float64)

    assert torch.allclose(new_p1, expected_p1, atol=1e-4), f"Test 1 权重错误: {new_p1}"
    assert torch.allclose(new_c1, expected_c1, atol=1e-4), f"Test 1 缓存错误: {new_c1}"

    # 核心安全测试：验证绝无原地修改 (In-place check)
    assert new_p1 is not p1 and new_c1 is not c1, "安全警告: 返回了原张量引用！"
    assert torch.allclose(p1, p1_orig) and torch.allclose(c1, c1_orig), "安全警告: 输入张量被原地篡改！"
    print(f"Test 1 通过 (官方标准示例与非原地修改验证): p={new_p1.item():.4f}, cache={new_c1.item():.4f}")

    # 测 2: 2D 多维张量矩阵并行更新
    p2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    g2 = torch.tensor([[0.2, 0.4], [0.6, 0.8]], dtype=torch.float64)
    c2 = torch.zeros_like(p2)
    new_p2, new_c2 = RMSProp_step(p2, g2, c2, lr=0.01, beta=0.9)

    expected_c2 = 0.1 * (g2 ** 2)
    expected_p2 = p2 - 0.01 * (g2 / (torch.sqrt(expected_c2) + 1e-8))

    assert new_p2.shape == p2.shape and new_c2.shape == c2.shape, "Test 2 形状不匹配"
    assert torch.allclose(new_c2, expected_c2), "Test 2 缓存计算错误"
    assert torch.allclose(new_p2, expected_p2), "Test 2 权重计算错误"
    print(f"Test 2 通过 (2D 张量并行计算与形状保持): shape={new_p2.shape}")

    # 测 3: 零梯度测试 (无梯度时权重保持不变，缓存按 beta 指数衰减)
    p3 = torch.tensor([5.0])
    g3 = torch.tensor([0.0])
    c3 = torch.tensor([1.0])
    new_p3, new_c3 = RMSProp_step(p3, g3, c3, lr=0.1, beta=0.9)
    assert torch.allclose(new_p3, p3), "Test 3 零梯度时权重发生变动"
    assert torch.allclose(new_c3, torch.tensor([0.9])), "Test 3 缓存衰减计算错误"
    print("Test 3 通过 (零梯度下缓存指数衰减，权重保持不变)")

    # 测 4: 形状不一致拦截
    try:
        RMSProp_step(torch.randn(2), torch.randn(3), torch.randn(2), lr=0.1, beta=0.9)
        raise RuntimeError("未能拦截形状不一致")
    except AssertionError:
        pass
    print("Test 4 通过 (形状不一致断言拦截正常)")

    print("\n[PASS] 所有 4 个测试用例全部完美通过！")


