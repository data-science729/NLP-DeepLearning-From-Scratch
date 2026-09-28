import torch
import torch.nn as nn
import math


# MHA 多头注意力机制
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        # 1. 确保 d_model 能被 n_heads 整除（比如 512 维分成 8 个头，每个头就是 64 维）
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // self.n_heads  # 每个头分到的维度 (例如: 512 // 8 = 64)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # 步骤 1: 通过线性层生成 Q K V
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # -------------------------------------------------------------
        # 步骤 2：拆分多头 (Reshape + Transpose)
        # 将 d_model 拆成 n_heads × head_dim，并将 n_heads 维度提到前面
        # 形状变化：[batch_size, seq_len, d_model]
        #       -> [batch_size, seq_len, n_heads, head_dim]
        #       -> [batch_size, n_heads, seq_len, head_dim]
        # -------------------------------------------------------------
        # 在执行 .view() 的时候，刚从线性层 self.w_q(q) 出来的张量 Q 在物理内存中是完全连续存储的。
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, value=-1e-9)

        attn_weights = self.softmax(scores)
        context = torch.matmul(attn_weights, V)

        # 把调换的维度调回来: [batch_size, seq_len_q, n_heads, head_dim]
        # 【关键】因为调用了 transpose，内存变得不连续，必须加 .contiguous() 才能进行 .view()
        # 把多头重新合并成 d_model 维: [batch_size, seq_len_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # -------------------------------------------------------------
        # 步骤 7：通过最终的线性层 W_o 进行多头特征融合输出
        # output 形状：[batch_size, seq_len_q, d_model]
        # -------------------------------------------------------------
        output = self.w_o(context)
        return output
