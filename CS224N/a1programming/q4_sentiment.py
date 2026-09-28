#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CS224n: 深度学习在自然语言处理中的应用
Assignment 1: 题目 4 - 情感分析 (Sentiment Analysis)

运行方式:
    1. 使用第 3 题手写训练的词向量:
       python q4_sentiment.py --yourvectors
    2. 使用官方预训练的 GloVe 词向量:
       python q4_sentiment.py --pretrained
"""

import argparse
import itertools
import numpy as np
import matplotlib
# 后台绘图引擎
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from utils.treebank import StanfordSentiment
import utils.glove as glove
from q3_sgd import load_saved_params


def getArguments():
    """解析命令行参数，互斥选择词向量来源。"""
    parser = argparse.ArgumentParser(description="CS224n Assignment 1 - 情感分类器")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pretrained", dest="pretrained", action="store_true",
                       help="使用预训练的 GloVe 50维词向量")
    group.add_argument("--yourvectors", dest="yourvectors", action="store_true",
                       help="使用在第3题手写训练的 10维 Word2Vec 词向量")
    return parser.parse_args()


def getSentenceFeatures(tokens, wordVectors, sentence):
    """4(a): 通过对句子中所有单词的词向量求平均，获取句子的特征表示向量。

    参数:
    tokens -- 单词映射到索引的词表字典 dict (如 tokens["good"] = 12)
    wordVectors -- 词向量矩阵，形状为 (V, D)
    sentence -- 句子中的单词字符串列表 list of str

    返回值:
    sentVector -- 句子的特征向量，形状为 (D,)
    """
    sentVector = np.zeros((wordVectors.shape[1],))

    ### YOUR CODE HERE: 实现句子特征向量提取 (平均池化)
    # 1. 列表推导式: 获取当前句子中所有单词在词表字典中的索引列表
    indices = [tokens[w] for w in sentence if w in tokens]

    # 2. NumPy 向量化: 一次性切片提取这批词的词向量 (形状为 N x D)
    #    并在轴 0 (样本/词维度) 上直接求平均值，获得整句话的综合特征表示 (形状为 D,)
    if len(indices) > 0:
        sentVector = np.mean(wordVectors[indices], axis=0)
    ### END YOUR CODE

    assert sentVector.shape == (wordVectors.shape[1],)
    return sentVector


def getRegularizationValues():
    """4(c): 超参数搜索 - 返回待评估的一组候选正则化系数 lambda 列表。

    返回值:
    values -- 从小到大排好序的浮点数列表
    """
    values = None   # 在下方代码块中为 values 赋值

    ### YOUR CODE HERE: 设置候选正则化参数列表
    # 使用半个数量级 (步长 0.5) 在 10^-4 到 10^2 范围内等比采样 (共 13 个候选值)
    # 覆盖从极弱正则化 (0.0001) 到极强正则化 (100.0) 的全范围
    values = [float(10.0 ** x) for x in np.arange(-4.0, 2.5, 0.5)]
    ### END YOUR CODE

    return sorted(values)


def chooseBestModel(results):
    """4(c): 依据验证集 (dev set) 上的准确率，从候选结果中挑选最佳模型。

    参数:
    results -- 包含各模型测试表现的字典列表，格式如下:
        [
            {
                "reg": regularization_value,
                "clf": trained_classifier,
                "train": trainAccuracy,
                "dev": devAccuracy,
                "test": testAccuracy
            },
            ...
        ]

    返回值:
    bestResult -- 选出的字典对象 (验证集 devAccuracy 最高的那个)
    """
    bestResult = None

    ### YOUR CODE HERE: 选出在 dev 验证集上准确率最高的字典项
    # 使用 Python 内置的 max 函数结合 lambda 表达式，以字典中的 "dev" 准确率为 key 选取最优模型
    bestResult = max(results, key=lambda res: res["dev"])
    ### END YOUR CODE

    return bestResult


def accuracy(y, yhat):
    """计算分类准确率 (百分比)。"""
    assert y.shape == yhat.shape
    return np.sum(y == yhat) * 100.0 / y.size


def plotRegResults(results):
    """4(e): 绘制训练集与验证集准确率随正则化系数变化的半对数折线图。"""
    regs = [r["reg"] for r in results]
    train_acc = [r["train"] for r in results]
    dev_acc = [r["dev"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.semilogx(regs, train_acc, label="Train Accuracy", marker='o')
    plt.semilogx(regs, dev_acc, label="Dev Accuracy", marker='s')
    plt.xlabel("Regularization Parameter (lambda)")
    plt.ylabel("Accuracy (%)")
    plt.title("Classification Accuracy vs. Regularization")
    plt.legend()
    plt.grid(True)
    plt.savefig("q4_reg_acc.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("正则化准确率曲线已保存至: q4_reg_acc.png")


def outputConfusionMatrix(clf, features, labels, sentences,
                          out_img="q4_dev_conf.png", out_txt="q4_dev_pred.txt"):
    """4(f)(g): 绘制混淆矩阵并导出预测结果明细供错误归因分析。"""
    preds = clf.predict(features)
    cm = confusion_matrix(labels, preds)

    # 1. 绘制并保存混淆矩阵图像 (q4_dev_conf.png)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Dev Set)")
    plt.colorbar()
    tick_marks = np.arange(5)
    classes = ["--", "-", "0", "+", "++"]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    plt.close()
    print("混淆矩阵图已保存至: %s" % out_img)

    # 2. 导出预测对比明细文本 (q4_dev_pred.txt)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Index\tTrue\tPred\tSentence\n")
        for idx in range(len(labels)):
            sent_str = " ".join(sentences[idx])
            f.write(f"{idx}\t{labels[idx]}\t{preds[idx]}\t{sent_str}\n")
    print("错误分析文本已保存至: %s" % out_txt)


def main():
    args = getArguments()

    # 1. 加载数据集
    print("正在加载斯坦福情感树库 (SST) 数据集...")
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    # 2. 加载对应的词向量
    if args.yourvectors:
        print("\n--> 使用在 Q3 训练完成的 10 维 Word2Vec 词向量...")
        _, wordVectors0, _ = load_saved_params()
        if wordVectors0 is None:
            raise FileNotFoundError("未找到 Q3 训练的词向量检查点 saved_params_*.npy，请先确认 Q3 是否已训练完成。")
        # 将输入词向量与输出词向量相加作为最终特征词向量 (标准做法)
        wordVectors = (wordVectors0[:nWords, :] + wordVectors0[nWords:, :])
    elif args.pretrained:
        print("\n--> 使用官方预训练的 50 维 GloVe 词向量...")
        wordVectors = glove.loadWordVectors(tokens)

    dimVectors = wordVectors.shape[1]
    print(f"词向量矩阵加载成功！形状: {wordVectors.shape}")

    # 3. 提取训练集、验证集、测试集特征
    print("\n正在对训练集、验证集、测试集提取句子平均特征向量...")
    trainset = dataset.getTrainSentences()
    nTrain = len(trainset)
    trainFeatures = np.zeros((nTrain, dimVectors))
    trainLabels = np.zeros((nTrain,), dtype=np.int32)
    trainSentences = []
    for i in range(nTrain):
        words, trainLabels[i] = trainset[i]
        trainFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)
        trainSentences.append(words)

    devset = dataset.getDevSentences()
    nDev = len(devset)
    devFeatures = np.zeros((nDev, dimVectors))
    devLabels = np.zeros((nDev,), dtype=np.int32)
    devSentences = []
    for i in range(nDev):
        words, devLabels[i] = devset[i]
        devFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)
        devSentences.append(words)

    testset = dataset.getTestSentences()
    nTest = len(testset)
    testFeatures = np.zeros((nTest, dimVectors))
    testLabels = np.zeros((nTest,), dtype=np.int32)
    testSentences = []
    for i in range(nTest):
        words, testLabels[i] = testset[i]
        testFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)
        testSentences.append(words)

    print(f"特征提取完成！样本数: 训练集 {nTrain}, 验证集 {nDev}, 测试集 {nTest}")

    # 4. 网格搜索不同正则化系数 lambda
    regValues = getRegularizationValues()
    print(f"\n开始网格搜索正则化超参数，待测列表 ({len(regValues)} 个): {regValues}")
    results = []
    for reg in regValues:
        print(f"  正在评估 lambda = {reg:g} ...", end="", flush=True)
        # sklearn 的 C 为正则化强度的倒数: C = 1 / lambda
        clf = LogisticRegression(
            C=1.0 / (reg + 1e-12),
            solver='lbfgs',
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
        clf.fit(trainFeatures, trainLabels)

        trainPred = clf.predict(trainFeatures)
        devPred = clf.predict(devFeatures)
        testPred = clf.predict(testFeatures)

        trainAcc = accuracy(trainLabels, trainPred)
        devAcc = accuracy(devLabels, devPred)
        testAcc = accuracy(testLabels, testPred)

        print(f" -> Train: {trainAcc:.2f}%, Dev: {devAcc:.2f}%, Test: {testAcc:.2f}%")

        results.append({
            "reg": reg,
            "clf": clf,
            "train": trainAcc,
            "dev": devAcc,
            "test": testAcc
        })

    # 5. 选择在验证集上表现最好的最佳模型
    best = chooseBestModel(results)
    print("\n" + "=" * 55)
    print("【网格搜索最佳模型评估结果】")
    print(f"最佳正则化系数 (Best Regularization): {best['reg']}")
    print(f"训练集准确率 (Train Accuracy):        {best['train']:.2f}%")
    print(f"验证集准确率 (Dev Accuracy):          {best['dev']:.2f}%")
    print(f"测试集准确率 (Test Accuracy):         {best['test']:.2f}%")
    print("=" * 55 + "\n")

    # 6. 生成可视化分析图表与明细数据
    plotRegResults(results)
    outputConfusionMatrix(best["clf"], devFeatures, devLabels, devSentences)


if __name__ == "__main__":
    main()
