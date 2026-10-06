"""
题目：按值进行梯度裁剪（Gradient Clipping by Value）
难度：简单（Easy）
分类：深度学习（Deep Learning）

题目描述：
    实现一个按值进行梯度裁剪的函数。该技术常用于神经网络训练中，
    通过将梯度值限制在指定区间内来防止梯度爆炸（Exploding Gradients）。

    给定一个梯度的 NumPy 数组 gradients 以及一个截断阈值 clip_value，
    将数组中的每个梯度元素截断至 [-clip_value, clip_value] 区间内：
    - 任何大于 clip_value 的数值均应被置为 clip_value；
    - 任何小于 -clip_value 的数值均应被置为 -clip_value；
    - 处于该区间内的数值保持不变。

    该函数应支持任意形状（如 1D、2D 等）的梯度数组，
    并返回与输入形状完全一致的裁剪后梯度数组。

参数：
    gradients (np.ndarray): 输入的梯度数组，支持任意维度与形状。
    clip_value (float): 截断阈值，梯度将被限制在 [-clip_value, clip_value] 之间。

返回：
    np.ndarray: 裁剪后的梯度数组，形状与输入 gradients 一致。

示例：
    输入：
        gradients = [1.0, -2.0, 3.0, -0.5, 0.2], clip_value = 1.5
    输出：
        [1.0, -1.5, 1.5, -0.5, 0.2]
    解释：
        梯度值 1.0 在 [-1.5, 1.5] 范围内，保持不变；
        数值 -2.0 小于 -1.5，被裁剪为 -1.5；
        数值 3.0 大于 1.5，被裁剪为 1.5；
        数值 -0.5 与 0.2 均在有效区间内，保持不变。
"""
import torch


def clip_by_value(gradients: torch.Tensor, clip_value: float) -> torch.Tensor:
    assert isinstance(clip_value, (int, float)) and clip_value >= 0
    gradients = torch.as_tensor(gradients, dtype=torch.float64)

    return torch.clamp(gradients, min=-float(clip_value), max=float(clip_value))


if __name__ == "__main__":
    # 用例 1: Deep-ML 官方标准 1D 示例
    g1 = [1.0, -2.0, 3.0, -0.5, 0.2]
    c1 = 1.5
    expected1 = torch.tensor([1.0, -1.5, 1.5, -0.5, 0.2], dtype=torch.float64)
    out1 = clip_by_value(g1, c1)
    assert torch.allclose(out1, expected1), "Test 1 Failed"
    print(f"Test 1 Passed: {out1.tolist()}")

    # 用例 2: 2D 矩阵高维保持形状测试
    g2 = torch.tensor([[10.0, -20.0], [-0.1, 0.8]], dtype=torch.float64)
    c2 = 1.0
    expected2 = torch.tensor([[1.0, -1.0], [-0.1, 0.8]], dtype=torch.float64)
    out2 = clip_by_value(g2, c2)
    assert out2.shape == g2.shape, "Shape mismatch in Test 2"
    assert torch.allclose(out2, expected2), "Test 2 Failed"
    print(f"Test 2 Passed (2D Shape preserved): {out2.shape}")

    # 用例 3: 边界情况 clip_value = 0 (全部截断归零)
    g3 = torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float64)
    out3 = clip_by_value(g3, 0.0)
    assert torch.allclose(out3, torch.zeros_like(g3)), "Test 3 Failed"
    print(f"Test 3 Passed (Zero clipping): {out3.tolist()}")

    # 用例 4: 阈值极大 (无截断，原样保留)
    g4 = torch.randn(3, 4, dtype=torch.float64)
    out4 = clip_by_value(g4, 1e5)
    assert torch.allclose(out4, g4), "Test 4 Failed"
    print("Test 4 Passed (Large threshold preserves input)")

    # 用例 5: 防御性断言测试 (传入非法负数必须抛出异常)
    try:
        clip_by_value(g1, -1.0)
        raise RuntimeError("Assertion check failed to catch negative clip_value")
    except AssertionError:
        print("Test 5 Passed (AssertionError correctly raised for negative threshold)")

    print("\n[PASS] 所有测试用例全部通过！")



