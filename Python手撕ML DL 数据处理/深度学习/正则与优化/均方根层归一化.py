# 实现均方根层归一化 (RMSNorm)
# 难度：简单
# 领域：深度学习
#
# 实现均方根层归一化（RMSNorm），这是一种用于现代 Transformer 架构（如 LLaMA 和 T5）中的归一化技术。
# 与标准的 Layer Normalization 不同，RMSNorm 不通过减去均值来对激活值进行中心化。
# 相反，它仅使用均方根统计量进行归一化，在保持竞争力的性能的同时，使计算更加简单。
#
# 你的函数应接受以下参数：
# - x: 一个形状为 (batch_size, features) 的二维 NumPy 数组，表示输入激活值。
# - g: 一个形状为 (features,) 的一维 NumPy 数组，表示可学习的增益（缩放）参数。->不用reshape
# - eps: 一个用于数值稳定性的微小浮点数（默认值为 1e-5）。
#
# 该函数应在特征维度上独立计算每个样本（行）的均方根（RMS）并进行归一化，然后乘以增益参数 g 进行缩放。
# 返回与 x 形状相同的 NumPy 数组作为归一化和缩放后的结果。
#
# 示例：
# 输入：
# x = np.array([[1.0, 2.0, 3.0]])
# g = np.array([1.0, 1.0, 1.0])
# eps = 1e-5
#
# 输出：
# [[0.4629, 0.9258, 1.3887]]
#
# 推导步骤：
# 第 1 步：计算平方值：[1, 4, 9]。
# 第 2 步：计算平方值的均值：(1 + 4 + 9) / 3 = 4.6667。
# 第 3 步：加上 epsilon 并开平方：RMS = sqrt(4.6667 + 1e-5) = 2.1602。
# 第 4 步：将每个元素除以 RMS：[1/2.1602, 2/2.1602, 3/2.1602] = [0.4629, 0.9258, 1.3887]。
# 第 5 步：乘以增益（此处全为 1，因此值保持不变）。

import torch
def rms_norm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64)
    g = torch.as_tensor(g, dtype=torch.float64)
    assert x.ndim == 2
    batch_size, features = x.shape
    assert g.numel() == features

    # 1. 沿特征维度 (dim=-1) 计算平方均值，形状保持为 (batch_size, 1)
    mean_sq = (x ** 2).mean(dim=-1, keepdim=True)

    # 2. 计算均方根 RMS
    rms = torch.sqrt(mean_sq + eps)

    # 3. 归一化并乘以增益 g (g 保持一维，自动沿特征维度广播)
    x_rms = x / rms
    y = x_rms * g
    return y


if __name__ == "__main__":
    # 测试用例 1: 官方示例 (batch_size=1, features=3)
    x = [[1.0, 2.0, 3.0]]
    g = [1.0, 1.0, 1.0]
    eps = 1e-5

    expected = torch.tensor([[0.4629, 0.9258, 1.3887]], dtype=torch.float64)
    res = rms_norm(x, g, eps)

    assert torch.allclose(res, expected, atol=1e-4), "测试用例 1 未通过！"
    print("[PASS] 测试用例 1 通过: 官方示例精确匹配！")
    print("输出结果:\n", res)

    # 测试用例 2: 多批次广播测试 (batch_size=2 != features=3，防范广播隐患)
    x_multi = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]
    g_multi = [1.0, 2.0, 3.0]
    res_multi = rms_norm(x_multi, g_multi)
    assert res_multi.shape == (2, 3), "多批次输出形状错误！"
    print("[PASS] 测试用例 2 通过: 多批次广播形状校验成功 (2, 3)！")








