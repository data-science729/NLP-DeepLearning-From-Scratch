"""
1376. Leaky ReLU
难度: 简单

【题目描述】
LeakyReLU 在正半轴等同恒等映射，负半轴保留一小段斜率，减轻“神经元死亡”。

【实现要求】
- z > 0 时返回 z，否则返回 αz
- α 由参数传入

【算法定义】
f(z) = {
    z,   z > 0
    αz,  z <= 0
}

【输出与判定】
返回 Python 浮点标量；浮点比较：绝对误差 <= 10^-5 或相对误差 <= 10^-5（满足其一即通过）。

【参数说明】
- z: float，输入标量
- alpha: float，负半轴斜率

【返回值】
- float，激活值

【示例 1】
输入：
    z = 0
    alpha = 0.01
输出：
    返回值 = 0.0

【限制条件】
- |z| <= 10^4
- 0 < α <= 1
"""
"""
1376. Leaky ReLU
难度: 简单
"""
"""
1376. Leaky ReLU
难度: 简单
"""

def leaky_relu(z: float, alpha: float) -> float:
    # 纯标量输入：直接条件判断
    if z > 0:
        return float(z)
    else:
        return float(alpha * z)

# 也可以简写为一行 Python 三元表达式：
# def leaky_relu(z: float, alpha: float) -> float:
#     return float(z if z > 0 else alpha * z)


if __name__ == '__main__':
    print("=" * 60)
    print("测试用例 1: 零点边界 (示例 1)")
    print(f"输出: {leaky_relu(0.0, 0.01)} | 预期: 0.0")
    assert abs(leaky_relu(0.0, 0.01) - 0.0) < 1e-5, "用例 1 失败"

    print("\n测试用例 2: 正半轴恒等映射")
    print(f"输出: {leaky_relu(5.5, 0.01)} | 预期: 5.5")
    assert abs(leaky_relu(5.5, 0.01) - 5.5) < 1e-5, "用例 2 失败"

    print("\n测试用例 3: 负半轴微小斜率")
    print(f"输出: {leaky_relu(-2.0, 0.01)} | 预期: -0.02")
    assert abs(leaky_relu(-2.0, 0.01) - (-0.02)) < 1e-5, "用例 3 失败"

    print("\n>>> 判定: 全部通过！<<<")
    print("=" * 60)



