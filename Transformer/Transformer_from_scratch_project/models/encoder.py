import torch.nn as nn
from .embedding import TransformerEmbedding
from .layers import EncoderLayer


# Encoder 编码器整体: 一个 Embedding 层 + N 个重复堆叠的 EncoderLayer 组成的整体
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_len, device, dropout=0.1):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size, d_model, max_len, device, dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        # 将输入的词 ID 转换为带有位置信息的 Embedding 向量
        out = self.emb(x)
        # 依次穿过 N 个 EncoderLayer 进行深度特征提取
        for layer in self.layers:
            out = layer(out, mask)

        return out
