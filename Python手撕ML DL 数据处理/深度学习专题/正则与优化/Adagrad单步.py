"""
题目：1407. Adagrad 优化步（Adagrad Optimization Step）
难度：简单（Easy）
分类：优化算法（Optimization）/ 深度学习

题目描述：
    实现自适应梯度算法（Adagrad）的单步参数更新逻辑。
    Adagrad 通过累积历史梯度平方和来为各个参数动态调整学习率：
    梯度频繁出现的特征对应较小的更新步长，稀疏特征对应较大的更新步长。

    在单步更新中，给定当前参数 param（θ）、当前梯度 grad（g）、
    历史梯度平方累积 cache（G）、学习率 lr 以及防止除以零的微小常数 eps（ε），
    依次完成累积量更新与参数更新，并返回更新后的参数与新累积值组成的二元组。

核心公式：
    1. 累积梯度平方更新：
       G_{new} = G + g²
    2. 参数更新：
       θ_{new} = θ - lr * (g / (sqrt(G_{new}) + ε))

参数：
    param (float): 当前参数值 θ。
    grad (float): 当前步的梯度 g。
    cache (float): 历史梯度平方累积值 G。
    lr (float): 学习率（步长），满足 lr > 0。
    eps (float): 数值稳定项 ε（防止分母为 0），满足 eps > 0。

返回：
    Tuple[float, float]:
        包含 (新参数值 θ_{new}, 新累积梯度平方和 G_{new}) 的元组。

评测标准：
    返回结果与标准答案的浮点误差满足：
    绝对误差 <= 1e-5 或相对误差 <= 1e-5 即可判定通过。

示例 1：
    输入：
        param = 1, grad = 0.5, cache = 1, lr = 0.01, eps = 1e-8
    输出：
        (0.9955278640850004, 1.25)
    解释：
        1. 更新梯度平方累积：
           G_{new} = 1 + 0.5² = 1 + 0.25 = 1.25
        2. 计算分母有效步长：
           sqrt(G_{new}) + eps = sqrt(1.25) + 1e-8
                               ≈ 1.11803398875 + 1e-8
                               ≈ 1.11803399875
        3. 计算参数调整量：
           delta = 0.01 * (0.5 / 1.11803399875) ≈ 0.0044721359149996
        4. 更新参数：
           θ_{new} = 1 - 0.0044721359149996 ≈ 0.9955278640850004

限制条件：
    - 测例为标量参数输入
    - lr > 0
    - eps > 0
"""


import torch
import math
def adagrad_step(param:float,grad:float,cache:float,lr:float,eps:float=1e-8)->tuple[float,float]:
    assert cache>=0 and lr>=0 and eps>0
    new_cache = cache+(grad**2)
    eff_lr = lr/(math.sqrt(new_cache)+eps)
    new_param = param - eff_lr*grad
    return new_param,new_cache


if __name__ == "__main__":
    print("=" * 50)
    print("开始执行【1407. Adagrad 优化步】测试套件")
    print("=" * 50)

    # 测 1: 题目官方标准示例
    p1, g1, c1, lr1, eps1 = 1, 0.5, 1, 0.01, 1e-8
    new_p, new_c = adagrad_step(p1, g1, c1, lr1, eps1)

    expected_p = 0.9955278640850004
    expected_c = 1.25

    assert abs(new_p - expected_p) < 1e-5, f"参数错误: {new_p}"
    assert abs(new_c - expected_c) < 1e-5, f"累积量错误: {new_c}"
    print(f"Test 1 通过 (官方标准示例): new_param={new_p:.6f}, new_cache={new_c:.4f}")

    # 测 2: 梯度为 0（无推力，cache 不变，参数不变）
    p2, g2, c2 = 2.0, 0.0, 4.0
    new_p2, new_c2 = adagrad_step(p2, g2, c2, lr=0.1)
    assert new_p2 == 2.0 and new_c2 == 4.0
    print("Test 2 通过 (零梯度静止验证)")

    # 测 3: 学习率自适应衰减性验证（梯度越大，等效步长被压制得越小）
    # 当梯度极大时，sqrt(new_cache) 很大，更新量不会爆炸
    _, c_large = adagrad_step(0.0, 100.0, 0.0, lr=0.1)
    assert c_large == 10000.0
    print("Test 3 通过 (大梯度平方累积验证)")

    print("\n[PASS] 所有测试全部通过！")




