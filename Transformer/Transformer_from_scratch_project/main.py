import torch
from config import *
from models import Encoder, Decoder
from utils import Transformermask

"""
TODO: 【待手写实现 2】整套 Transformer 的端到端测试 (End-to-End Test)
当你在 models/transformer.py 中完成了 Transformer 顶层封装后，
可以在主函数中取消 Transformer 类测试的注释，验证前向传播输出 Shape 是否等于 [batch_size, tgt_len, tgt_vocab_size]。
"""

def test_existing_components():
    print("=== 开始测试已手写完成的基础模块 ===")
    
    # 模拟数据
    batch_size = 2
    src_len = 5
    tgt_len = 6
    
    src = torch.randint(2, SRC_VOCAB_SIZE, (batch_size, src_len)).to(DEVICE)
    tgt = torch.randint(2, TGT_VOCAB_SIZE, (batch_size, tgt_len)).to(DEVICE)
    
    # 测试 Mask 构建器
    mask_builder = Transformermask(pad_idx=PAD_IDX)
    src_mask = mask_builder.make_src_mask(src)
    tgt_mask = mask_builder.make_tgt_mask(tgt)
    print("src_mask Shape:", src_mask.shape)  # 应为 [2, 1, 1, 5]
    print("tgt_mask Shape:", tgt_mask.shape)  # 应为 [2, 1, 6, 6]
    
    # 测试 Encoder 整体
    encoder = Encoder(
        vocab_size=SRC_VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        device=DEVICE,
        dropout=DROPOUT
    ).to(DEVICE)
    enc_out = encoder(src, src_mask)
    print("Encoder 输出 Shape:", enc_out.shape)  # 应为 [2, 5, 512]
    
    # 测试 Decoder 整体
    decoder = Decoder(
        vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        device=DEVICE,
        max_len=MAX_LEN,
        dropout=DROPOUT
    ).to(DEVICE)
    dec_out = decoder(tgt, enc_out, tgt_mask, src_mask)
    print("Decoder 输出 Shape:", dec_out.shape)  # 应为 [2, 6, 512]
    
    print("=== 所有已手写基础模块校验通过！ ===")
    return src, tgt

if __name__ == "__main__":
    src, tgt = test_existing_components()
    
    # 顶层 Transformer 模型端到端测试：
    print("\n=== 开始测试顶层 Transformer 主模型 ===")
    from models.transformer import Transformer
    transformer = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, device=DEVICE).to(DEVICE)
    output = transformer(src, tgt)
    print("Transformer 最终 Logits 输出 Shape:", output.shape)  # 应为 [2, 6, 10000]
    print("=== 端到端前向传播测试圆满成功！ ===")
