"""
1555. 二维最大池化 (简单)

【题目描述】
对二维特征图做 k×k 最大池化，窗口不重叠，步长等于 k。

【实现要求】
- 输入的高 (H) 和宽 (W) 均可被 k 整除。
- 每个窗口取最大值，窗口互不重叠。
- 输出尺寸为：(H / k) × (W / k)。

【计算公式】
对输出格 (i, j)：
    Y[i, j] = max_{0 <= u, v < k} X[i * k + u, j * k + v]

【伪代码】
    for i in 0 .. H/k - 1:
        for j in 0 .. W/k - 1:
            Y[i, j] = max(X[i*k : (i+1)*k, j*k : (j+1)*k])
    return Y

【判定标准】
返回二维矩阵；浮点比较：绝对误差 <= 1e-5 或相对误差 <= 1e-5（满足其一即通过）。

【示例 1】
输入：
    x = [
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9,  10, 11, 12],
        [13, 14, 15, 16]
    ]
    k = 2
输出：
    [
        [6.0, 8.0],
        [14.0, 16.0]
    ]
解释：
    窗口大小为 2x2，步长为 2：
    - 左上窗口 max([1, 2, 5, 6]) = 6.0
    - 右上窗口 max([3, 4, 7, 8]) = 8.0
    - 左下窗口 max([9, 10, 13, 14]) = 14.0
    - 右下窗口 max([11, 12, 15, 16]) = 16.0

【限制条件】
- 输入为二维矩阵，H, W <= 64
- k >= 1，且 H, W 均可被 k 整除
"""
import torch


def max_pool2d(x: torch.Tensor, k: int) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64)
    H, W = x.shape

    # 极简断言校验 (维度、正数、整除)
    assert x.ndim == 2 and k >= 1
    assert H % k == 0 and W % k == 0

    # 计算输出尺寸
    out_H = H // k
    out_W = W // k
    Y = torch.zeros((out_H, out_W), dtype=torch.float64)

    # 滑动窗口取最大值
    for i in range(out_H):
        for j in range(out_W):
            window = x[i * k : (i + 1) * k, j * k : (j + 1) * k]
            Y[i, j] = window.max()

    return Y


if __name__ == "__main__":
    # 测试用例 1: 官方示例 (4x4, k=2)
    x = [
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9,  10, 11, 12],
        [13, 14, 15, 16]
    ]
    res = max_pool2d(x, k=2)
    expected = torch.tensor([
        [6.0, 8.0],
        [14.0, 16.0]
    ], dtype=torch.float64)
    assert torch.allclose(res, expected, atol=1e-5), "测试用例 1 未通过！"
    print("[PASS] 测试用例 1 通过: 官方示例")

    # 测试用例 2: 鲁棒性整除异常拦截测试
    try:
        max_pool2d([[1, 2, 3], [4, 5, 6]], k=2)  # H=2 能整除，但 W=3 不能整除
        print("[FAIL] 未能正确拦截非整除异常！")
    except AssertionError:
        print("[PASS] 测试用例 2 通过: 成功拦截非整除断言")

    # 测试用例 3: 最小边界 k=1
    small_x = [[3.5, 4.2], [1.1, 9.9]]
    res_k1 = max_pool2d(small_x, k=1)
    assert torch.allclose(res_k1, torch.tensor(small_x, dtype=torch.float64)), "测试用例 3 未通过！"
    print("[PASS] 测试用例 3 通过: 边界 k=1 测试")
