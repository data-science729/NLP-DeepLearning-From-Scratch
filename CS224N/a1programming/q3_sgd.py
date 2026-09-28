#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS224n: 深度学习在自然语言处理中的应用
Assignment 1: 题目 3(f) - 随机梯度下降算法 (Stochastic Gradient Descent, SGD)

本文件实现了通用的 SGD 优化器，包含以下核心机制：
1. 参数迭代更新 (θ = θ - α * ∇J(θ))
2. 后处理/约束映射机制 (postprocessing，如针对 word2vec 词向量的每步行归一化)
3. 学习率退火衰减 (Learning Rate Annealing: 每隔固定步数学习率减半)
4. 指数移动平均损失平滑 (Exponential Moving Average, EMA)
5. 检查点保存与断点恢复 (Checkpointing: 应对大规模词向量耗时训练的断点续训)
"""

import glob
import os.path as op
import pickle
import random
import numpy as np

# 容错与断点保护机制：每隔多少次 SGD 迭代保存一次参数检查点
SAVE_PARAMS_EVERY = 5000


def load_saved_params():
    """检查点恢复辅助函数。
    
    扫描当前目录下的检查点文件（匹配 saved_params_*.npy），
    找到迭代次数最大的最新检查点，并恢复训练参数以及随机数生成器的状态，
    以支持训练中断后的断点继续训练。

    返回值:
    st -- 恢复的起始迭代次数（若无保存文件则返回 0）
    params -- 保存的模型参数（若无保存文件则返回 None）
    state -- 保存的 random 模块随机状态元组（若无保存文件则返回 None）
    """
    st = 0
    # 扫描匹配模式的所有检查点文件
    for f in glob.glob("saved_params_*.npy"):
        iter_num = int(op.splitext(op.basename(f))[0].split("_")[2])
        if iter_num > st:
            st = iter_num

    if st > 0:
        # Python 3 中 pickle 读写必须采用二进制模式 ("rb")
        with open("saved_params_%d.npy" % st, "rb") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None


def save_params(iter_num, params):
    """保存参数检查点的辅助函数。

    参数:
    iter_num -- 当前迭代轮次编号（整型）
    params -- 需要持久化保存的模型参数对象（如词向量矩阵等）
    """
    # Python 3 中 pickle 必须采用二进制写入模式 ("wb")
    with open("saved_params_%d.npy" % iter_num, "wb") as f:
        pickle.dump(params, f)
        # 同步保存 random 状态，保证随机采样的连续性和完全可复现性
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False, PRINT_EVERY=10):
    """随机梯度下降优化算法 (Stochastic Gradient Descent, SGD)。

    参数:
    f -- 待优化的目标函数。接受单个参数 x，并返回一个二元组 (cost, grad)：
         - cost: 标量损失值
         - grad: 关于参数 x 的梯度，形状与 x 完全一致
    x0 -- 优化的初始参数点（NumPy ndarray 或标量）
    step -- 初始学习率 / 步长 (Learning Rate / Step Size, α)
    iterations -- SGD 运行的总迭代步数
    postprocessing -- 可选的参数后处理函数。在每次梯度更新后调用，
                      用于对参数施加约束（例如在 word2vec 中需要对词向量进行模长归一化）；
                      若不指定，则默认直接返回参数本身
    useSaved -- 布尔值。若为 True，则优先尝试从本地已有的检查点文件恢复训练
    PRINT_EVERY -- 打印训练进度的频率（每隔多少次迭代输出一次平滑后的损失）

    返回值:
    x -- 经过 iterations 次迭代优化后的最终参数
    """
    # 学习率衰减周期：每隔 20,000 轮将学习率减半（模拟退火机制）
    ANNEAL_EVERY = 20000

    # 1. 断点恢复逻辑
    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            # 补偿之前已经历轮次所对应的学习率衰减
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)
        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    # 如果未提供后处理函数，则定义为恒等映射
    if not postprocessing:
        postprocessing = lambda x: x

    exp_cost = None

    # 2. SGD 核心优化主循环
    for iter_num in range(start_iter + 1, iterations + 1):
        cost = None

        ### TODO: 在此实现你的 SGD 核心单步更新逻辑
        # 提示：
        # 1. 调用目标函数 f(x) 获取当前参数下的标量损失 cost 与梯度 grad
        # 2. 根据梯度下降公式更新参数：x = x - step * grad
        # 3. 对更新后的参数执行后处理约束：x = postprocessing(x)
        
        cost, grad = f(x)
        x -= step * grad
        x = postprocessing(x)

        ### END TODO

        # 3. 打印进度：采用指数加权移动平均 (EMA) 平滑随机梯度的剧烈波动
        if iter_num % PRINT_EVERY == 0:
            if not exp_cost:
                exp_cost = cost
            else:
                exp_cost = 0.95 * exp_cost + 0.05 * cost
            print("iter %d: %f" % (iter_num, exp_cost))

        # 4. 定期保存检查点（启用 useSaved 时有效）
        if iter_num % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter_num, x)

        # 5. 学习率退火：每隔 ANNEAL_EVERY 轮迭代，步长减半
        if iter_num % ANNEAL_EVERY == 0:
            step *= 0.5

    return x


def sanity_check():
    """基础健全性测试。
    
    使用简单的二次凸函数 f(x) = sum(x^2) 检验 SGD 的收敛性。
    最小值点在 x = 0 处，梯度为 2 * x。
    验证在不同初始值（正数、零、负数）下 SGD 是否均能收敛至 0 附近。
    """
    quad = lambda x: (np.sum(x ** 2), x * 2.0)

    print("=" * 50)
    print("正在运行 SGD 基础健全性测试 (Sanity Check)...")
    print("=" * 50)

    print("\n[测试 1] 初始点 x0 = 0.5:")
    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print("测试 1 优化收敛结果:", t1)
    assert abs(t1) <= 1e-6, "测试 1 失败：x 未能收敛至 0 附近"

    print("\n[测试 2] 初始点 x0 = 0.0 (极值点处):")
    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print("测试 2 优化收敛结果:", t2)
    assert abs(t2) <= 1e-6, "测试 2 失败：x 偏离了极值点 0"

    print("\n[测试 3] 初始点 x0 = -1.5 (负数初始值):")
    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print("测试 3 优化收敛结果:", t3)
    assert abs(t3) <= 1e-6, "测试 3 失败：x 未能收敛至 0 附近"

    print("\n" + "-" * 50)
    print("恭喜！所有 SGD 基础健全性测试均已通过！")
    print("-" * 50 + "\n")


def your_sgd_test():
    """自定义扩展测试函数。
    
    你可以在此添加更多的测试用例（如多维矩阵优化、带 postprocessing 约束的优化等）。
    直接运行：
        python q3_sgd.py
    该函数不会计入作业评分。
    """
    print("正在运行自定义扩展测试...")
    ### TODO: 在此编写你的自定义测试（可选）


    pass

    ### END TODO
    print("自定义测试完成！\n")


if __name__ == "__main__":
    sanity_check()
    your_sgd_test()
