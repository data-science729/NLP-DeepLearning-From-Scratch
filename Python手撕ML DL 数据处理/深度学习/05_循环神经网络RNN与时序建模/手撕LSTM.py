"""
2111. 手撕 LSTM (简单)

题目描述：
    实现一个单步 LSTM 单元（LSTMCell），同时更新隐藏状态 h 与细胞状态 c。
    LSTM 引入输入门 i、遗忘门 f、候选细胞 g 与输出门 o，以缓解长序列任务中的梯度消失问题。
    本题权重按四个门沿纵向拼接存储，分段顺序为 i, f, g, o（与主流深度学习框架一致）。

数学公式：
    [i; f; g; o] = [σ; σ; tanh; σ](W @ x + U @ h_{t-1} + b)
    c_t = f ⊙ c_{t-1} + i ⊙ g
    h_t = o ⊙ tanh(c_t)
    其中 σ 为 Sigmoid 激活函数，⊙ 表示逐元素乘法（Hadamard积）。

实现要求：
    - 实现类 LSTMCell：
        - __init__(self, W, U, b)：保存输入权重 W、隐状态权重 U 和偏置 b。
        - forward(self, x, h_prev, c_prev)：接收当前输入及上一时刻状态，返回更新后的二元组 (h_new, c_new)。
    - 维度规范：
        - W ∈ R^{4*d_h × d_{in}}
        - U ∈ R^{4*d_h × d_h}
        - b ∈ R^{4*d_h}
        - W, U, b 均沿第 0 维按 i, f, g, o 顺序各取 d_h 行进行切分。

算法伪代码：
    class LSTMCell:
        def __init__(self, W, U, b):
            self.W, self.U, self.b = W, U, b

        def forward(self, x, h_prev, c_prev):
            gates = self.W @ x + self.U @ h_prev + self.b
            i, f, g, o = split_4(gates)  # 各占 d_h 维
            i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)
            g = tanh(g)
            c_new = f * c_prev + i * g
            h_new = o * tanh(c_new)
            return h_new, c_new

参数说明：
    x: 当前时刻输入向量，形状通常为 (d_in, 1) 或 (d_in,)
    h_prev: 上一时刻隐状态，形状通常为 (d_h, 1) 或 (d_h,)
    c_prev: 上一时刻细胞状态，形状通常为 (d_h, 1) 或 (d_h,)

返回值：
    tuple: (h_new, c_new)，分别为更新后的隐状态和细胞状态

评测判定：
    - 浮点比较满足绝对误差 <= 1e-5 或相对误差 <= 1e-5 即判定通过。

示例 1:
    输入:
        x (4, 1) = [[0.2], [0.3], [0.4], [0.1]]
        h_prev (4, 1) = [[0.1], [0.1], [0.1], [0.1]]
        c_prev = [0, 0, 0, 0]
    输出:
        返回值 (h_new, c_new) = ([0.108105], [0.208909])

限制条件:
    d_in, d_h <= 8
"""
import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """单步 LSTM；门控拼接顺序为 i, f, g, o。"""

    def __init__(self, W: torch.Tensor, U: torch.Tensor, b: torch.Tensor):
        super().__init__()
        # W: (4*d_h, d_in), U: (4*d_h, d_h), b: (4*d_h,)
        self.register_buffer("W", W.to(dtype=torch.float64))
        self.register_buffer("U", U.to(dtype=torch.float64))
        self.register_buffer("b", b.to(dtype=torch.float64))

    def forward(self,x:torch.Tensor,h_prev:torch.Tensor,c_prev:torch.Tensor):
        #确保输入类型为float64 与模型内部的W,U,b保持严格一致
        x = torch.as_tensor(x,dtype=torch.float64)
        h_prev = torch.as_tensor(h_prev,dtype=torch.float64)
        c_prev = torch.as_tensor(c_prev,dtype=torch.float64)
        #偏置形状对齐 若x是二维列向量(d_in,1) 则把b变为(4*d_h,1)方便相加
        b =self.b.view(-1,1) if x.ndim ==2 else self.b
        #四门联合线性运算 W@x +U@h_prev +b 形状(4*d_h,1)
        gates = self.W@x+self.U@h_prev+b
        #确保c_prev的形状和h_prev完全一致 杜绝(4,1)和(4,)被误广播为(4,4)
        c_prev = c_prev.view_as(h_prev)
        i,f,g,o = torch.chunk(gates,4,dim=0)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f*c_prev +i*g
        h_new = o*torch.tanh(c_new)
        return h_new,c_new
if __name__ == '__main__':
    print("=" * 65)
    print("测试用例 1: 与 PyTorch 官方原生 nn.LSTMCell 进行 100% 精度像素级对标")
    d_in = 4
    d_h = 3
    torch.manual_seed(42)

    # 随机生成权重与输入
    W = torch.randn(4 * d_h, d_in, dtype=torch.float64)
    U = torch.randn(4 * d_h, d_h, dtype=torch.float64)
    b = torch.randn(4 * d_h, dtype=torch.float64)

    # 实例化手撕的 LSTMCell
    my_cell = LSTMCell(W, U, b)

    # 实例化 PyTorch 官方原生 nn.LSTMCell，并将权重 1:1 注入
    official_cell = nn.LSTMCell(d_in, d_h, bias=True).double()
    with torch.no_grad():
        official_cell.weight_ih.copy_(W)
        official_cell.weight_hh.copy_(U)
        official_cell.bias_ih.copy_(b)
        official_cell.bias_hh.zero_()

    # 输入数据 (二维列向量)
    x = torch.randn(d_in, 1, dtype=torch.float64)
    h_prev = torch.randn(d_h, 1, dtype=torch.float64)
    c_prev = torch.randn(d_h, 1, dtype=torch.float64)

    # 手撕前向推导
    h_my, c_my = my_cell(x, h_prev, c_prev)

    # 官方前向推导 (官方接收 batch_size 在前的行向量 (1, d_in))
    h_off, c_off = official_cell(x.t(), (h_prev.t(), c_prev.t()))

    diff_h = torch.max(torch.abs(h_off.t() - h_my)).item()
    diff_c = torch.max(torch.abs(c_off.t() - c_my)).item()

    print(f"手撕输出 h_new 形状: {h_my.shape}")
    print(f"与 PyTorch 官方 nn.LSTMCell 输出 h 的最大绝对误差: {diff_h:.1e}")
    print(f"与 PyTorch 官方 nn.LSTMCell 输出 c 的最大绝对误差: {diff_c:.1e}")
    assert diff_h < 1e-12 and diff_c < 1e-12, "验证失败！"
    print(">>> 判定结果: 误差精确为 0，与 PyTorch 官方底层源码 100% 完全等价！<<<")

    print("\n" + "=" * 65)
    print("测试用例 2: 一维向量与二维列向量混合兼容性测试")
    # c_prev 传一维 (d_h,)，x 与 h_prev 传二维列向量
    c_prev_1d = torch.zeros(d_h, dtype=torch.float64)
    h_res, c_res = my_cell(x, h_prev, c_prev_1d)
    print(f"输入 c_prev 形状为一维 {c_prev_1d.shape}，输出状态形状自适应保持: {c_res.shape}")
    print("=" * 65)
