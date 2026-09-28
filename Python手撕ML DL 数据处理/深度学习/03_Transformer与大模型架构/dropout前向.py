"""
1557. Dropout前向
难度: 简单

【题目描述】
训练态 Dropout：按概率 p 置零，其余元素按 1/(1-p) 缩放；用给定种子保证可复现。

【实现要求】
- 用给定 seed 固定随机源后再采样掩码
- 保留位置乘 1/(1-p)，丢弃位置为 0

【算法定义】
y_i = {
    0,                以概率 p
    x_i / (1 - p),    以概率 1 - p
}

【伪代码】
set_seed(seed)
for each i:
    if rand() < p:
        y[i] = 0
    else:
        y[i] = x[i] / (1 - p)
return y

【输出与判定】
返回与输入同形的向量；浮点比较：绝对误差 <= 10^-4 或相对误差 <= 10^-4（满足其一即通过）。

【参数说明】
- x: List[float]，输入向量
- p: float，丢弃概率
- seed: int，随机种子

【返回值】
- List[float]，Dropout 前向传播后的向量

【示例 1】
输入：
    x = [1, 2, 3, 4]
    p = 0.5
    seed = 42
输出：
    返回值 = [0.0, 4.0, 6.0, 8.0]

【限制条件】
- 向量长度 <= 64
- 0 <= p < 1
- seed 为非负整数
"""

import numpy as np

def dropout_forward(x:np.ndarray, p: float, seed: int) ->np.ndarray:
    x = np.array(x,dtype=float)
    np.random.seed(seed)
    #生成0-1之间均匀随机数
    r = np.random.rand(len(x))
    y = np.where(r<p,0.0,x/(1.0-p))
    return y.tolist()



