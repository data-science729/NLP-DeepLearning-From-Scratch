import torch
import torch.nn as nn
from .attention import MultiheadAttention


# LayerNorm 归一化层
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=10e-12):
        super(LayerNorm, self).__init__()
        # 1. 可学习的缩放参数 gamma，初始化全为 1（与词向量维度 d_model 相同）
        self.gamma = nn.Parameter(torch.ones(d_model))
        # 2. 可学习的平移参数 beta，初始化全为 0
        self.beta = nn.Parameter(torch.zeros(d_model))
        # 3. 防止分母为 0 的极小常数 (epsilon)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        return out


# FFN 前馈神经网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)


# 编码器单层 EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiheadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.mha(x, x, x, mask)
        x1 = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x1)
        x2 = self.norm2(x1 + self.dropout2(ffn_output))
        return x2


# 解码器单层 DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.msa = MultiheadAttention(d_model, n_heads)
        self.cross_attn = MultiheadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, dec_x, enc_out, tgt_mask=None, src_mask=None):
        """
        参数说明：
        dec_x: 解码器当前的输入 Tensor [batch_size, tgt_len, d_model]
        enc_out: 编码器 Encoder 的最终输出 Tensor [batch_size, src_len, d_model]
        tgt_mask: 目标序列掩码 (Causal Mask 因果掩码)，防止偷看未来词
        src_mask: 源序列掩码 (Padding Mask)，屏蔽编码器填充符号
        """
        # 子层 1: 掩码自注意力
        self_attn_out = self.msa(q=dec_x, k=dec_x, v=dec_x, mask=tgt_mask)
        x1 = self.norm1(dec_x + self.dropout1(self_attn_out))

        # 子层 2: 交叉注意力层，Q 来自解码器上一层的输出 x1, K, V 来自编码器 Encoder 的输出 enc_out
        cross_attn_out = self.cross_attn(q=x1, k=enc_out, v=enc_out, mask=src_mask)
        x2 = self.norm2(x1 + self.dropout2(cross_attn_out))

        # 子层 3: FFN
        ffn_out = self.ffn(x2)
        x3 = self.norm3(x2 + self.dropout3(ffn_out))
        return x3
