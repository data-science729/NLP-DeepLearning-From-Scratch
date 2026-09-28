import torch.nn as nn
from .embedding import TransformerEmbedding
from .layers import DecoderLayer


# Decoder 解码器整体: 一个 Embedding 层 + N 个重复堆叠的 DecoderLayer 组成的整体
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, device, max_len, dropout=0.1):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size, d_model, max_len, device, dropout)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
            ]
        )

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        out = self.emb(x)
        for layer in self.layers:
            out = layer(out, enc_out, tgt_mask, src_mask)
        return out
