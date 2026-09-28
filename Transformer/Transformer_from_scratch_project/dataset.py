import torch
from torch.utils.data import Dataset, DataLoader

"""
TODO: 【待手写实现 - 数据加载模块】
任务说明：
1. 实现文本处理与 Vocab（词表映射，如 word2idx / idx2word）。
2. 自定义 Dataset 类（继承自 torch.utils.data.Dataset）：
   - __init__: 接收原始语料，完成文本清洗与数值 TokenID 转换
   - __len__: 返回数据集样本数量
   - __getitem__: 返回单个 (src_seq, tgt_seq) 样本索引点
3. 实现 collate_fn 完成动态 Padding 填充。
4. 构建 DataLoader 用于批量 Batch 加载。
"""

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        super().__init__()
        # TODO: 请在此处实现数据读取与预处理
        pass

    def __len__(self):
        # TODO: 返回数据集长度
        return 0

    def __getitem__(self, idx):
        # TODO: 返回单个样本数据
        pass
