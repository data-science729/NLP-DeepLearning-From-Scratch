import torch

# 统一超参数配置文件 与Transformer原始论文保持一致
# TODO: 可根据实际需求调整这些模型与训练参数

# 模型维度参数
D_MODEL = 512       # 词向量及隐藏层维度
N_HEADS = 8         # 多头注意力头数
D_FF = 2048         # 前馈神经网络隐藏层维度
NUM_LAYERS = 6      # Encoder 与 Decoder 的堆叠层数
DROPOUT = 0.1       # Dropout 概率
MAX_LEN = 5000      # 位置编码的最大序列长度

# 词表参数
SRC_VOCAB_SIZE = 10000  # 源语言词表大小 (示例值)
TGT_VOCAB_SIZE = 10000  # 目标语言词表大小 (示例值)
PAD_IDX = 1             # 填充 Token 的 ID 索引

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = torch.device('cuda')
