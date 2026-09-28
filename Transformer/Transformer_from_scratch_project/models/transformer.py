import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from utils.mask import Transformermask

"""
TODO: 【待手写实现 1】Transformer 顶层主模型封装
任务说明：
1. 实例化核心组件：
   - 实例化 Encoder 模块
   - 实例化 Decoder 模块
   - 实例化输出词表映射线性层：self.projection = nn.Linear(d_model, tgt_vocab_size)
   - 实例化掩码工具类：self.mask_builder = Transformermask(pad_idx=pad_idx)

2. 实现 forward(self, src, tgt) 方法：
   a. 生成 mask:
      src_mask = self.mask_builder.make_src_mask(src)
      tgt_mask = self.mask_builder.make_tgt_mask(tgt)
   b. 执行编码器前向传播:
      enc_out = self.encoder(src, src_mask)
   c. 执行解码器前向传播:
      dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
   d. 映射输出概率 logits:
      output = self.projection(dec_out)
   e. 返回 output (形状应为 [batch_size, tgt_len, tgt_vocab_size])
"""

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, d_ff=2048,
                 num_layers=6, max_len=5000, pad_idx=1, device='cuda', dropout=0.1):
        super().__init__()
        # 1. 掩码构造工具
        self.mask_builder = Transformermask(pad_idx=pad_idx)
        # 2. 编码器 Encoder
        self.encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, num_layers, max_len, device, dropout)
        # 3. 解码器 Decoder
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads, d_ff, num_layers, device, max_len, dropout)
        # 4. 输出预测映射线性层
        self.projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # 步骤 1: 自动生成源序列 Mask 与目标序列 Mask
        src_mask = self.mask_builder.make_src_mask(src)
        tgt_mask = self.mask_builder.make_tgt_mask(tgt)
        # 步骤 2: Encoder 前向传播
        enc_out = self.encoder(src, src_mask)
        # 步骤 3: Decoder 前向传播 (含 Cross-Attention)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        # 步骤 4: 映射到预测 Logits
        output = self.projection(dec_out)
        return output
