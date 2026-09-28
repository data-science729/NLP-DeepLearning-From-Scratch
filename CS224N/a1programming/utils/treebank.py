#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS224n: 深度学习在自然语言处理中的应用
Assignment 1: 斯坦福情感树库 (Stanford Sentiment Treebank, SST) 数据加载与预处理工具模块
已全面适配 Python 3。
"""

import os
import pickle
import random
import numpy as np


class StanfordSentiment:
    """斯坦福情感树库 (SST) 数据集封装类。
    
    提供词汇表构建、负采样概率表生成、上下文滑动窗口采样以及情感标签映射功能。
    """
    def __init__(self, path=None, tablesize=1000000):
        if not path:
            # 自动探测数据集路径，支持在项目任意目录下被导入和调用
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidate_path = os.path.join(base_dir, "datasets", "stanfordSentimentTreebank")
            if os.path.exists(candidate_path):
                path = candidate_path
            else:
                path = "utils/datasets/stanfordSentimentTreebank"

        self.path = path
        self.tablesize = tablesize

    def tokens(self):
        """获取词表字典与词频统计信息。"""
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0

        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if w not in tokens:
                    tokens[w] = idx
                    revtokens.append(w)
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        tokens["UNK"] = idx
        revtokens.append("UNK")
        tokenfreq["UNK"] = 1
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens

    def sentences(self):
        """读取并分词语料库中的所有句子（转换为小写）。"""
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences

        sentences = []
        sentence_file = os.path.join(self.path, "datasetSentences.txt")
        with open(sentence_file, "r", encoding="utf-8", errors="replace") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split()[1:]
                # 转换为小写单词列表
                sentences.append([w.lower() for w in splitted])

        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)

        return self._sentences

    def numSentences(self):
        """语料库中总句子数。"""
        if hasattr(self, "_numSentences") and self._numSentences:
            return self._numSentences
        else:
            self._numSentences = len(self.sentences())
            return self._numSentences

    def allSentences(self):
        """带下采样（Subsampling）丢弃高频词后的句子列表。"""
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        sentences = self.sentences()
        rejectProb = self.rejectProb()
        tokens = self.tokens()
        allsentences = [[w for w in s
                         if 0 >= rejectProb[tokens[w]] or random.random() >= rejectProb[tokens[w]]]
                        for s in sentences * 30]

        allsentences = [s for s in allsentences if len(s) > 1]

        self._allsentences = allsentences
        return self._allsentences

    def getRandomContext(self, C=5):
        """从语料库中随机抽取一个中心词以及其在窗口大小 C 内的上下文词列表。"""
        allsent = self.allSentences()
        sentID = random.randint(0, len(allsent) - 1)
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1)

        context = sent[max(0, wordID - C):wordID]
        if wordID + 1 < len(sent):
            context += sent[wordID + 1:min(len(sent), wordID + C + 1)]

        centerword = sent[wordID]
        context = [w for w in context if w != centerword]

        if len(context) > 0:
            return centerword, context
        else:
            return self.getRandomContext(C)

    def sent_labels(self):
        """加载所有句子的连续情感分值（0.0 ~ 1.0）。"""
        if hasattr(self, "_sent_labels") and self._sent_labels:
            return self._sent_labels

        dictionary = dict()
        phrases = 0
        with open(os.path.join(self.path, "dictionary.txt"), "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                splitted = line.split("|")
                dictionary[splitted[0].lower()] = int(splitted[1])
                phrases += 1

        labels = [0.0] * phrases
        with open(os.path.join(self.path, "sentiment_labels.txt"), "r", encoding="utf-8", errors="replace") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                line = line.strip()
                if not line:
                    continue
                splitted = line.split("|")
                labels[int(splitted[0])] = float(splitted[1])

        sent_labels = [0.0] * self.numSentences()
        sentences = self.sentences()
        for i in range(self.numSentences()):
            sentence = sentences[i]
            full_sent = " ".join(sentence).replace('-lrb-', '(').replace('-rrb-', ')')
            if full_sent in dictionary:
                sent_labels[i] = labels[dictionary[full_sent]]
            else:
                sent_labels[i] = 0.5  # 默认中性

        self._sent_labels = sent_labels
        return self._sent_labels

    def dataset_split(self):
        """加载官方的训练集(0)、测试集(1)、验证集(2)划分索引。"""
        if hasattr(self, "_split") and self._split:
            return self._split

        split = [[] for _ in range(3)]
        with open(os.path.join(self.path, "datasetSplit.txt"), "r", encoding="utf-8", errors="replace") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split(",")
                split[int(splitted[1]) - 1].append(int(splitted[0]) - 1)

        self._split = split
        return self._split

    def getRandomTrainSentence(self):
        """随机获取一条训练集句子及其对应的五分类情感类别。"""
        split = self.dataset_split()
        sentId = split[0][random.randint(0, len(split[0]) - 1)]
        return self.sentences()[sentId], self.categorify(self.sent_labels()[sentId])

    def categorify(self, label):
        """将 0~1 的连续情感得分映射为 5 个离散类别 (0~4)。"""
        if label <= 0.2:
            return 0
        elif label <= 0.4:
            return 1
        elif label <= 0.6:
            return 2
        elif label <= 0.8:
            return 3
        else:
            return 4

    def getDevSentences(self):
        """获取验证集数据。"""
        return self.getSplitSentences(2)

    def getTestSentences(self):
        """获取测试集数据。"""
        return self.getSplitSentences(1)

    def getTrainSentences(self):
        """获取训练集数据。"""
        return self.getSplitSentences(0)

    def getSplitSentences(self, split=0):
        """根据划分标识获取对应的数据集句子与标签列表。"""
        ds_split = self.dataset_split()
        return [(self.sentences()[i], self.categorify(self.sent_labels()[i])) for i in ds_split[split]]

    def sampleTable(self):
        """构建用于高效负采样的单模分布轮盘表（基于 word_freq ^ 0.75）。"""
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable

        nTokens = len(self.tokens())
        samplingFreq = np.zeros((nTokens,))
        self.allSentences()
        for i in range(nTokens):
            w = self._revtokens[i]
            if w in self._tokenfreq:
                freq = 1.0 * self._tokenfreq[w]
                # Mikolov 经典负采样 3/4 次幂平滑
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq

        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize

        self._sampleTable = [0] * self.tablesize

        j = 0
        for i in range(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j

        return self._sampleTable

    def rejectProb(self):
        """计算高频词随机丢弃的丢弃概率表。"""
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb

        threshold = 1e-5 * self._wordcount
        nTokens = len(self.tokens())
        rejectProb = np.zeros((nTokens,))
        for i in range(nTokens):
            w = self._revtokens[i]
            freq = 1.0 * self._tokenfreq[w]
            # 经典重加权下采样概率
            rejectProb[i] = max(0.0, 1.0 - np.sqrt(threshold / freq))

        self._rejectProb = rejectProb
        return self._rejectProb

    def sampleTokenIdx(self):
        """在负采样轮盘表中以 O(1) 复杂度随机抽取一个负样本词的全局索引。"""
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]
