import torch
import torch.nn as nn
import torch.optim as optim
from config import *

"""
TODO: 【待手写实现 3.1】模型训练与验证逻辑
任务说明：
1. 损失函数：使用带 ignore_index=PAD_IDX 的交叉熵 Loss
   criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
2. 优化器：使用 Adam 优化器，设置 betas=(0.9, 0.98), eps=1e-9
3. 动态学习率调度器：根据论文实现 Warmup 增长而后按倒平方根衰减的 Noam LR Scheduler
4. 单 Epoch 训练循环逻辑：
   - 提取输入: src_seq, tgt_seq
   - 变形对齐: tgt_input = tgt_seq[:, :-1], tgt_y = tgt_seq[:, 1:]
   - 前向传播并计算 Loss
   - 反向传播 optimizer.zero_grad() -> loss.backward() -> optimizer.step()
"""

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    # TODO: 编写训练 Loop 逻辑
    return total_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    # TODO: 编写验证 Loop 逻辑
    return total_loss
