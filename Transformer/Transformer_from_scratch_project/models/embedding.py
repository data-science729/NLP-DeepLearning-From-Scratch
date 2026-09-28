import torch
import torch.nn as nn
import math
from torch import Tensor


# 将输入的离散单词数字 ID，转换为高维的连续语义向量（Dense Vector），并按照 Transformer 论文要求进行数值缩放。
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model, padding_idx=1)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        # 重写 forward，按论文要求乘以 sqrt(d_model)
        return super().forward(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)    # Shape: [max_len, 1]

        _2i = torch.arange(0, d_model, step=2, device=device).float()          # Shape: [d_model // 2]
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))  # 偶数列用 sin
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))   # 奇数列用 cos

    # 根据当前输入的句子实际长度（seq_len），从事先生成好的大位置编码表（self.encoding）中，截取出刚好够用的前 seq_len 行位置向量返回。
    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, device, dropout):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)
