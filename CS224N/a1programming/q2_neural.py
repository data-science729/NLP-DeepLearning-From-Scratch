import random
import numpy as np

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """单隐层神经网络的前向传播与反向传播。

    计算交叉熵损失以及损失关于网络所有参数的梯度。

    参数:
    data -- 输入数据矩阵，形状为 (N, Dx)，其中 N 为样本数，Dx 为特征维度。
    labels -- 真实标签的 One-Hot 矩阵，形状为 (N, Dy)，Dy 为类别数。
    params -- 包含网络所有参数的扁平化一维数组。
    dimensions -- 网络维度列表 [Dx, H, Dy]，分别代表输入维度、隐层神经元数、输出类别数。

    返回值:
    cost -- 神经网络在当前数据上的交叉熵标量损失。
    grad -- 损失关于参数 params 的梯度，形状与 params 完全一致的扁平化一维数组。
    """
    ### 1. 解包网络参数（Unpack Parameters）
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    # 第一层权重 W1 与偏置 b1
    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H

    # 第二层权重 W2 与偏置 b2
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### 2. 前向传播（Forward Pass）
    # (1) 隐层线性变换: z1 = data * W1 + b1, 形状为 (N, H)
    z1 = np.dot(data, W1) + b1
    # (2) 隐层非线性激活: h = sigmoid(z1), 形状为 (N, H)
    h = sigmoid(z1)
    # (3) 输出层线性变换: z2 = h * W2 + b2, 形状为 (N, Dy)
    z2 = np.dot(h, W2) + b2
    # (4) 输出层概率归一化: y_hat = softmax(z2), 形状为 (N, Dy)
    y_hat = softmax(z2)

    # (5) 交叉熵损失计算: J = -sum(labels * log(y_hat))
    # 加上 1e-15 防止 log(0) 导致数值异常
    cost = -np.sum(labels * np.log(y_hat + 1e-15))

    ### 3. 反向传播（Backward Pass）
    # (1) 输出层误差: delta2 = y_hat - labels, 形状为 (N, Dy)
    delta2 = y_hat - labels

    # (2) 第二层参数梯度:
    # gradW2 = h^T * delta2, 形状为 (H, Dy)
    gradW2 = np.dot(h.T, delta2)
    # gradb2 沿批次维度 (axis=0) 求和, 形状为 (1, Dy)
    gradb2 = np.sum(delta2, axis=0, keepdims=True)

    # (3) 误差回传至隐层: dh = delta2 * W2^T, 形状为 (N, H)
    dh = np.dot(delta2, W2.T)

    # (4) 穿过 Sigmoid 激活层: delta1 = dh ⊙ sigmoid_grad(h), 形状为 (N, H)
    delta1 = dh * sigmoid_grad(h)

    # (5) 第一层参数梯度:
    # gradW1 = data^T * delta1, 形状为 (Dx, H)
    gradW1 = np.dot(data.T, delta1)
    # gradb1 沿批次维度 (axis=0) 求和, 形状为 (1, H)
    gradb1 = np.sum(delta1, axis=0, keepdims=True)

    ### 4. 展平并拼接梯度（Pack Gradients）
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    基础健全性检查。
    生成小型伪数据与随机参数，利用 gradcheck_naive 对反向传播梯度进行数值检验。
    """
    print("正在运行神经网络健全性检查...")
    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1.0

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] +
                             (dimensions[1] + 1) * dimensions[2], )

    passed = gradcheck_naive(lambda p: forward_backward_prop(data, labels, p, dimensions), params)
    assert passed, "神经网络健全性检查未通过！"
    print("健全性检查顺利通过！\n")


def your_sanity_checks():
    """
    自定义扩展测试（例如测试不同样本量、单样本极值、不同隐层大小等）。
    直接运行：
        python q2_neural.py
    该函数不会计入作业评分。
    """
    print("正在运行自定义测试...")
    # 测试单样本 (N=1) 和不同尺寸 [4, 8, 3]
    N = 1
    dimensions = [4, 8, 3]
    data = np.random.randn(N, dimensions[0])
    labels = np.zeros((N, dimensions[2]))
    labels[0, 1] = 1.0
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] +
                             (dimensions[1] + 1) * dimensions[2], )

    passed = gradcheck_naive(lambda p: forward_backward_prop(data, labels, p, dimensions), params)
    assert passed, "单样本测试未通过！"
    print("所有自定义测试均已通过！\n")


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
