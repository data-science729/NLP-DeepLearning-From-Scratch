#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS224n: 深度学习在自然语言处理中的应用
Assignment 1: 预训练 GloVe 词向量加载辅助模块
"""

import os
import numpy as np

DEFAULT_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "datasets", "glove.6B.50d.txt"
)


def loadWordVectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50):
    """从文本文件中读取预训练的 GloVe 词向量并匹配词表。
    
    参数:
    tokens -- 单词到词表索引的字典映射 dict
    filepath -- glove.6B.50d.txt 文本文件路径
    dimensions -- 词向量维度 (默认为 50 维)
    
    返回值:
    wordVectors -- 形状为 (len(tokens), dimensions) 的词向量矩阵
    """
    wordVectors = np.zeros((len(tokens), dimensions))
    if not os.path.exists(filepath):
        # 兼容当前目录相对路径
        candidate = "utils/datasets/glove.6B.50d.txt"
        if os.path.exists(candidate):
            filepath = candidate
        else:
            raise FileNotFoundError(
                f"未找到预训练 GloVe 词向量文件: {filepath}\n"
                "请确保已下载并解压 glove.6B.50d.txt 到 utils/datasets/ 目录下。"
            )

    print(f"正在从 {filepath} 读取 GloVe 预训练词向量...")
    loaded_count = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token in tokens:
                wordVectors[tokens[token]] = np.array([float(x) for x in row[1:]])
                loaded_count += 1
    print(f"GloVe 词向量加载完成！命中词汇: {loaded_count} / {len(tokens)}")
    return wordVectors
