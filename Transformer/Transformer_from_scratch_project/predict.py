import torch
from config import *

"""
TODO: 【待手写实现 3.2】推理解码策略 (Inference / Generation)
任务说明：
1. 实现贪婪搜索解码 (Greedy Search Decode)：
   - 输入 src 得到 enc_out
   - 初始化 tgt 序列为包含 SOS_TOKEN 的张量
   - 循环一步一步自回归预测下一个词 ID，直至遇到 EOS_TOKEN 或达到最大生成长度
2. (可选拓展) 实现束搜索解码 (Beam Search Decode)：
   - 维持 Top-K 概率最大的 candidate 假设路径序列
"""

def greedy_decode(model, src, max_len, start_symbol, end_symbol, device):
    """
    自回归贪婪搜索解码函数示例存根
    """
    model.eval()
    # TODO: 编写自回归预测逻辑
    pass
