"""
1391. 二维卷积 (中等)

【题目描述】
对输入矩阵做二维卷积：卷积核在填充后的平面上按步长滑动，累加逐点乘积，无偏置。

【实现要求】
1. padding 为四周零填充的圈数。
2. stride 为卷积核滑动步长。
3. 输出矩阵的高宽计算公式：
   out_H = floor((H + 2 * padding - kh) / stride) + 1
   out_W = floor((W + 2 * padding - kw) / stride) + 1

【算法定义】
对每个输出位置 (i, j)：
    Y[i, j] = sum_{u=0}^{kh-1} sum_{v=0}^{kw-1} X_pad[i*s + u, j*s + v] * K[u, v]

【伪代码】
Xp = zero_pad(X, padding)
for i, j in output_grid(stride):
    Y[i, j] = sum(Xp[i*s : i*s+kh, j*s : j*s+kw] * K)
return Y

【输出与判定】
- 返回二维特征图（浮点数矩阵）。
- 浮点比较判定：绝对误差 <= 10^-5 或 相对误差 <= 10^-5（满足其一即通过）。

【示例 1】
输入：
    input_matrix = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ]
    kernel = [
        [1, 2],
        [3, -1]
    ]
    padding = 0
    stride = 1
输出：
    返回值 = [
        [16.0, 21.0, 26.0, 31.0],
        [41.0, 46.0, 51.0, 56.0],
        [66.0, 71.0, 76.0, 81.0],
        [91.0, 96.0, 101.0, 106.0]
    ]

【限制条件】
- 输入边长 <= 32
- 卷积核边长 <= 5
- padding >= 0
- stride >= 1
"""
import torch


def conv_2d(x: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64)
    kernel = torch.as_tensor(kernel, dtype=torch.float64)
    H, W = x.shape
    kh, kw = kernel.shape

    # 极简断言校验 (维度、非负/正数超参、尺寸容纳)
    assert x.ndim == 2 and kernel.ndim == 2
    assert stride >= 1 and padding >= 0
    assert H + 2 * padding >= kh and W + 2 * padding >= kw

    # 零填充 padding
    if padding > 0:
        x_pad = torch.zeros((H + 2 * padding, W + 2 * padding), dtype=x.dtype)
        x_pad[padding : padding + H, padding : padding + W] = x
    else:
        x_pad = x

    # 计算输出尺寸并初始化
    out_H = (H + 2 * padding - kh) // stride + 1
    out_W = (W + 2 * padding - kw) // stride + 1
    Y = torch.zeros((out_H, out_W), dtype=torch.float64)

    # 双层循环滑动窗口点积 (Hadamard 积求和)
    for i in range(out_H):
        for j in range(out_W):
            window = x_pad[i * stride : i * stride + kh, j * stride : j * stride + kw]
            Y[i, j] = (window * kernel).sum()

    return Y


if __name__ == "__main__":
    # 测试用例 1: 题目默认示例 (padding=0, stride=1)
    input_matrix = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ]
    kernel = [
        [1, 2],
        [3, -1]
    ]
    res1 = conv_2d(input_matrix, kernel, padding=0, stride=1)
    expected1 = torch.tensor([
        [16.0, 21.0, 26.0, 31.0],
        [41.0, 46.0, 51.0, 56.0],
        [66.0, 71.0, 76.0, 81.0],
        [91.0, 96.0, 101.0, 106.0]
    ], dtype=torch.float64)
    assert torch.allclose(res1, expected1, atol=1e-5), "测试用例 1 未通过！"
    print("[PASS] 测试用例 1 通过 (padding=0, stride=1)")

    # 测试用例 2: 验证 padding=1 与 stride=2 的组合场景
    res2 = conv_2d(input_matrix, kernel, padding=1, stride=2)
    assert res2.shape == (3, 3), "测试用例 2 形状不符！"
    print("[PASS] 测试用例 2 通过 (padding=1, stride=2)，输出形状:", res2.shape)

    # 测试用例 3: 鲁棒性校验测试 - 当核大于图像且无 padding 时应触发断言
    try:
        huge_kernel = torch.ones((6, 6))
        conv_2d(input_matrix, huge_kernel, padding=0, stride=1)
        print("[FAIL] 未能正确拦截尺寸超限异常！")
    except AssertionError:
        print("[PASS] 测试用例 3 通过: 成功拦截尺寸超限断言")
