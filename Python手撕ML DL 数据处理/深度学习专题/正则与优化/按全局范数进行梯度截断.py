"""
题目：按全局范数进行梯度裁剪（Gradient Clipping by Global Norm）
分类：深度学习（Deep Learning）

题目描述：
    实现一个按全局范数（Global Norm）进行梯度裁剪的函数。
    给定一个包含多个梯度数组的列表（代表不同参数对应的梯度张量）以及一个最大范数阈值 max_norm：
    1. 计算所有参数梯度拼接后的全局 L2 范数（Global L2 Norm）；
    2. 若该全局范数超过阈值 max_norm，则等比例缩小所有梯度，使得裁剪后的全局范数等于 max_norm；
    3. 若全局范数不超过 max_norm，则所有梯度保持不变；
    4. 返回裁剪后的梯度列表，并保持原有的结构与各数组形状不变。

参数：
    gradients (List[np.ndarray]): 梯度数组列表，每个元素对应不同网络参数的梯度，支持任意形状与维度。
    max_norm (float): 全局 L2 范数的最大允许阈值。

返回：
    List[np.ndarray]: 裁剪后的梯度数组列表，结构与各个数组的形状与输入完全一致。

示例：
    输入：
        gradients = [np.array([3.0, 4.0]), np.array([0.0, 0.0])], max_norm = 1.0
    输出：
        [array([0.6, 0.8]), array([0.0, 0.0])]
    解释：
        全局 L2 范数为 √(3.0² + 4.0² + 0.0² + 0.0²) = √25.0 = 5.0；
        因为 5.0 > 1.0，触发截断；
        缩放因子（Scaling Factor）为 1.0 / 5.0 = 0.2；
        各个梯度数组乘以 0.2：
        - [3.0 × 0.2, 4.0 × 0.2] = [0.6, 0.8]
        - [0.0 × 0.2, 0.0 × 0.2] = [0.0, 0.0]
"""
import torch
def clip_by_global_norm(gradients:list[torch.Tensor],max_norm:float)->list[torch.Tensor]:
    assert isinstance(max_norm,(int,float)) and max_norm>0
    assert isinstance(gradients,(list,tuple)) and len(gradients)>0
    grads = [torch.as_tensor(g,dtype=torch.float64) for g in gradients]
    total_sq = sum((g**2).sum() for g in grads)
    global_norm = torch.sqrt(total_sq)
    if global_norm>max_norm:
        scale = max_norm/global_norm
    else:
        scale = 1.0
    clipped_gradients = [g*scale for g in grads]
    return clipped_gradients
if __name__ == "__main__":
    print("=" * 50)
    print("开始执行【按全局范数梯度截断】测试套件")
    print("=" * 50)
    # 测 1: 题目官方示例（模长 5.0 > 1.0，触发截断）
    g1 = [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 0.0])]
    max_norm1 = 1.0
    out1 = clip_by_global_norm(g1, max_norm1)
    expected1_0 = torch.tensor([0.6, 0.8], dtype=torch.float64)
    expected1_1 = torch.tensor([0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(out1[0], expected1_0, atol=1e-5), "Test 1 失败: 张量 0 结果不匹配"
    assert torch.allclose(out1[1], expected1_1, atol=1e-5), "Test 1 失败: 张量 1 结果不匹配"
    print(f"Test 1 通过 (官方标准示例): {out1[0].tolist()}, {out1[1].tolist()}")
    # 测 2: 保持形状（多张量、混合 1D / 2D / 3D 异构形状）
    g2 = [
        torch.randn(5),              # 1D 偏置
        torch.randn(3, 4),           # 2D 权重矩阵
        torch.randn(2, 3, 4)         # 3D 卷积核 / 注意力头
    ]
    out2 = clip_by_global_norm(g2, max_norm=2.0)
    for orig, clipped in zip(g2, out2):
        assert orig.shape == clipped.shape, f"Shape mismatch: {orig.shape} vs {clipped.shape}"
    print(f"Test 2 通过 (多维异构形状完全保留): {[x.shape for x in out2]}")
    # 测 3: 未超出阈值（原样保留，不进行缩放）
    g3 = [torch.tensor([0.1, 0.2]), torch.tensor([0.3, -0.1])]
    out3 = clip_by_global_norm(g3, max_norm=10.0)
    for orig, clipped in zip(g3, out3):
        assert torch.allclose(orig.to(torch.float64), clipped), "Test 3 失败: 未超标梯度被意外修改"
    print("Test 3 通过 (未超阈值时梯度保持不变)")
    # 测 4: 边界情况（全 0 梯度，不能触发除以 0 崩溃）
    g4 = [torch.zeros(4), torch.zeros(2, 2)]
    out4 = clip_by_global_norm(g4, max_norm=1.0)
    for clipped in out4:
        assert torch.allclose(clipped, torch.zeros_like(clipped)), "Test 4 失败: 全零梯度处理错误"
    print("Test 4 通过 (全零梯度边界安全)")
    # 测 5: 数学性质验证（截断后的新全局范数严格等于 max_norm）
    g5 = [torch.randn(10, 10) * 100 for _ in range(3)]  # 极大梯度
    target_norm = 1.5
    out5 = clip_by_global_norm(g5, max_norm=target_norm)
    new_norm = torch.sqrt(sum((g ** 2).sum() for g in out5))
    assert torch.isclose(new_norm, torch.tensor(target_norm, dtype=torch.float64), atol=1e-4), "Test 5 失败: 截断后模长不等于 max_norm"
    print(f"Test 5 通过 (截断后总模长精准收敛为 max_norm = {new_norm.item():.4f})")
    # 测 6: 健壮性异常拦截（非法参数正确抛出 AssertionError）
    try:
        clip_by_global_norm(g1, max_norm=-1.0)
        raise RuntimeError("未能拦截负数 max_norm")
    except AssertionError:
        pass
    try:
        clip_by_global_norm([], max_norm=1.0)
        raise RuntimeError("未能拦截空列表")
    except AssertionError:
        pass
    print("Test 6 通过 (非法输入断言拦截成功)")
    print("\n[PASS] 所有 6 个测试用例全部通过！")







