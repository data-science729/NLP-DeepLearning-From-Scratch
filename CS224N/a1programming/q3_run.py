#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS224n: 深度学习在自然语言处理中的应用
Assignment 1: 题目 3(g) - 使用 SGD 与 Skip-gram 在斯坦福情感树库 (SST) 上训练并可视化词向量

运行方式:
    python q3_run.py
"""

import time
import random
import numpy as np
import matplotlib
# 使用非交互式后台绘图引擎，避免环境缺少 GUI 支持时崩溃
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.treebank import StanfordSentiment
from q3_word2vec import word2vec_sgd_wrapper, skipgram, negSamplingCostAndGradient
from q3_sgd import sgd

# 1. 固定随机数种子以确保实验完全可复现
random.seed(314)
print("正在加载斯坦福情感树库 (Stanford Sentiment Treebank) 数据集...")
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)
print("数据集加载完成！词汇表单词总量:", nWords)

# 2. 模型超参数设置
dimVectors = 10     # 词向量维度：本作业训练 10 维词向量
C = 5              # 上下文窗口大小：中心词左右各看 5 个词

# 重置随机数种子
random.seed(31415)
np.random.seed(9265)

startTime = time.time()

# 3. 参数初始化
# 前半部分矩阵 (nWords, dimVectors) 为输入词向量，后半部分为输出词向量
wordVectors = np.concatenate(
    ((np.random.rand(nWords, dimVectors) - 0.5) / dimVectors,
     np.zeros((nWords, dimVectors))),
    axis=0
)

print("\n" + "=" * 60)
print("开始使用 SGD 训练 Word2Vec (Skip-gram + 负采样)...")
print("默认总轮次: 40000 轮 (断点自动保存开启)")
print("=" * 60 + "\n")

# 4. 调用 SGD 优化器开始训练
# 提示：如果是首次调试代码，可将 40000 临时改为 500 进行 15 秒极速测试！
wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(
        skipgram, tokens, vec, dataset, C, negSamplingCostAndGradient
    ),
    wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10
)

# 注意：训练过程中不需要调用行归一化。这并非 bug，因为在训练过程中频繁归一化会损失向量模长的语义信息。
print("\n" + "-" * 60)
print("训练完成！健全性检查：收敛后的最终 loss 应该在 10 附近或 10 以下。")
print("总训练耗时: %d 秒" % (time.time() - startTime))
print("-" * 60 + "\n")

# 5. 整合输入词向量与输出词向量（取两者拼接或相加）
wordVectors = np.concatenate(
    (wordVectors[:nWords, :], wordVectors[nWords:, :]),
    axis=0
)

# 6. 选择用于二维可视化的特定测试词汇
visualizeWords = [
    "the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
    "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
    "annoying"
]

visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]

# 7. SVD 奇异值分解降维（将 10 维词向量投影到 2 维平面）
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U, S, V = np.linalg.svd(covariance)
coord = temp.dot(U[:, 0:2])

# 8. 使用 Matplotlib 绘制二维散点词云图
plt.figure(figsize=(10, 8))
for i in range(len(visualizeWords)):
    plt.text(
        coord[i, 0], coord[i, 1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1)
    )

plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

output_png = "q3_word_vectors.png"
plt.savefig(output_png, bbox_inches='tight', dpi=150)
print(f"词向量二维可视化图表已成功生成并保存至: {output_png}")
