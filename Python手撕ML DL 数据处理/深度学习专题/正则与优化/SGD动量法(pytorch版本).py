"""
题目：带动量的随机梯度下降更新步（SGD with Momentum Step）
难度：中等（Medium）
分类：PyTorch / 深度学习（Deep Learning）

题目描述：
    实现单步带动量（Momentum）的随机梯度下降（SGD）更新逻辑。
    给定当前网络权重参数 w、当前梯度 grad、累积速度（动量缓存）v、学习率 lr 以及动量系数 mu，
    根据以下公式进行更新：
        v_new = mu * v + grad
        w_new = w - lr * v_new

    编写函数 momentum_step(w, grad, v, lr, mu)，以元组 (w_new, v_new) 的形式返回 PyTorch 张量。
    要求：禁止对输入张量进行原地修改（In-place modification），必须返回全新的张量。

参数：
    w (torch.Tensor): 当前权重参数张量，支持任意维度。
    grad (torch.Tensor): 参数对应的梯度张量，形状与 w 一致。
    v (torch.Tensor): 上一步累积的速度（动量）张量，形状与 w 一致。
    lr (float): 学习率（Learning Rate）。
    mu (float): 动量衰减系数（Momentum Coefficient）。

返回：
    Tuple[torch.Tensor, torch.Tensor]:
        包含 (w_new, v_new) 的元组，两者均为计算得到的新张量。

示例：
    输入：
        w = torch.tensor([1.0, 2.0])
        grad = torch.tensor([0.1, 0.2])
        v = torch.tensor([0.0, 0.0])
        lr = 0.1
        mu = 0.9
    输出：
        (tensor([0.9900, 1.9800]), tensor([0.1000, 0.2000]))
    解释：
        1. 速度更新：v 初始为 0，因此 v_new = 0.9 * [0.0, 0.0] + [0.1, 0.2] = [0.1, 0.2]；
        2. 权重更新：w_new = [1.0, 2.0] - 0.1 * [0.1, 0.2] = [0.99, 1.98]。
"""
import torch

def momentum_step(w, grad, v, lr, mu):
    assert lr>0
    assert 0<=mu<1
    assert w.shape == grad.shape == v.shape   #"w, grad, v 形状必须一致"
    v_new = mu*v + grad
    w_new = w - lr*v_new
    return w_new,v_new
