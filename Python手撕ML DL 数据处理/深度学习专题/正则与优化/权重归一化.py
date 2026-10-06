"""
1392. 权重归一化 (Weight Normalization)
难度：中等

【题目描述】
对神经网络线性层（全连接层）的权重执行权重归一化（Weight Normalization）：
不同于对数据/激活值进行归一化的 BN 和 LN，权重归一化直接对模型的“权重矩阵”进行参数重构（Reparameterization）。
将每个神经元的权重向量 w 解耦为其“方向向量 v”与“标量模长 g（幅值）”的乘积：
w = (g / ||v||) * v

【实现要求】
1. 输入权重方向张量 v，形状为 (out_features, in_features)。
2. 幅值参数 g，形状为 (out_features,) 或可通过广播匹配 (out_features, 1) 的向量，
   表示每个输出神经元权重向量的目标 L2 范数（模长）。
3. 沿输入特征维（in_features，即 axis=-1 或 dim=1）计算每个输出行向量的 L2 范数：
   ||v_i|| = sqrt(sum_{j} v[i, j]^2)
4. 默认平滑项 eps = 1e-12（防止分母范数为 0）。
5. 最终计算重构后的权重矩阵 w，形状与 v 保持一致 (out_features, in_features)。

【算法定义】
对每个输出神经元 i (0 <= i < out_features)：
    norm_v[i] = sqrt(sum_{j=0}^{in_features-1} v[i, j]^2 + eps)
    w[i, :] = (g[i] / norm_v[i]) * v[i, :]

【输入与输出】
- 输入：
  - v: 形状为 (out_features, in_features) 的二维张量 / 数组，表示权重的方向。
  - g: 形状为 (out_features,) 的一维张量 / 数组，表示每个神经元权重的标量模长。
  - eps: float，默认 1e-12。
- 输出：
  - 返回与 v 同形的二维张量 w (out_features, in_features)。
  - 浮点比较判定：绝对误差 <= 1e-5 或相对误差 <= 1e-5（满足其一即通过）。

【示例 1】
输入：
    v = [
        [3.0, 4.0, 0.0],
        [1.0, 2.0, 2.0]
    ]
    g = [10.0, 6.0]
输出：
    返回值 = [
        [6.0, 8.0, 0.0],
        [2.0, 4.0, 4.0]
    ]
解释：
    - 第 0 行向量 [3, 4, 0] 的模长为 sqrt(3^2 + 4^2 + 0^2) = 5.0。
      目标模长 g[0] = 10.0，缩放倍数 = 10.0 / 5.0 = 2.0。
      重构后第 0 行为 [6.0, 8.0, 0.0]。
    - 第 1 行向量 [1, 2, 2] 的模长为 sqrt(1^2 + 2^2 + 2^2) = 3.0。
      目标模长 g[1] = 6.0，缩放倍数 = 6.0 / 3.0 = 2.0。
      重构后第 1 行为 [2.0, 4.0, 4.0]。

【限制条件】
- out_features, in_features <= 64
- g 长度等于 out_features，且每个元素 g > 0
"""
import torch


def weight_norm(v: torch.Tensor, g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    v = torch.as_tensor(v, dtype=torch.float64)
    g = torch.as_tensor(g, dtype=torch.float64)
    assert v.ndim == 2
    out_features, in_features = v.shape
    assert out_features<=64 and in_features<=64
    assert g.numel() == out_features and (g>0).all()
    #形状对齐为列向量 g (out_features,1) 适配2D广播
    g = g.reshape(out_features,1)
    #计算每行的L2范数 保持形状为(out_features,1)
    norm_v = torch.sqrt((v**2).sum(dim=-1,keepdim=True)+eps)
    # 5. 重构权重矩阵: w = (g / ||v||) * v
    w = (g/norm_v)*v
    return w
if __name__ == "__main__":
    # 测试用例 1: 勾股数整体验算
    v = [
        [3.0, 4.0, 0.0],
        [1.0, 2.0, 2.0]
    ]
    g = [10.0, 6.0]
    expected = torch.tensor([
        [6.0, 8.0, 0.0],
        [2.0, 4.0, 4.0]
    ], dtype=torch.float64)
    res = weight_norm(v, g)
    assert torch.allclose(res, expected, atol=1e-5), "测试用例 1 未通过！"
    print("[PASS] 测试用例 1 通过: 勾股数精确验证通过！")
    # 验证重构后每行向量的模长是否严格等于 g
    reconstructed_norms = torch.sqrt((res ** 2).sum(dim=-1))
    target_norms = torch.tensor([10.0, 6.0], dtype=torch.float64)
    assert torch.allclose(reconstructed_norms, target_norms, atol=1e-5), "模长校验失败！"
    print("[PASS] 测试用例 2 通过: 重构权重模长精确等于目标 g！")




