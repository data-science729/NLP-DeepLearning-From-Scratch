import random
import numpy as np

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """矩阵行归一化函数。

    对矩阵 x 的每一行进行 L2 归一化，使其每行的向量模长（欧几里得范数）为 1。

    参数:
    x -- 形状为 (N, D) 的二维 NumPy 矩阵。

    返回值:
    x -- 归一化后的矩阵，形状与输入相同。
    """
    ### TODO: 在此实现行归一化
    denom = np.sqrt(np.sum(x**2,axis=1,keepdims=True))+1e-30
    x=x/denom


    ### END TODO
    return x


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """基于 Softmax 的 Word2Vec 代价与梯度计算函数。

    对应理论推导 3(a) 与 3(b)。

    参数:
    predicted -- 中心词的输入词向量 vc，形状为 (D,)
    target -- 真实期望的上下文目标词在词表中的整型索引 o
    outputVectors -- 所有单词的输出词向量矩阵 U，形状为 (V, D)
    dataset -- 数据集对象（在 Softmax 中未用到，为统一接口保留）

    返回值:
    cost -- 交叉熵标量损失 J
    gradPred -- 损失关于预测输入向量 vc 的梯度，形状为 (D,)
    grad -- 损失关于所有输出词向量矩阵 U 的梯度，形状为 (V, D)
    """
    ### TODO: 在此实现 Softmax 代价与梯度
    scores = np.dot(outputVectors,predicted)  #(V,)
    probs = softmax(scores)  #(V,)
    cost = -np.log(probs[target])  #R
    delta = probs.copy()  #(V,)
    delta[target]-=1.0   #delta的值只需要probs在target位置减去1即可,one-hot
    gradPred = np.dot(delta,outputVectors)  #内积，(D,)
    grad = np.outer(delta,predicted)   #(V,D)



    return cost, gradPred, grad

def getNegativeSamples(target,dataset,K):
    """从数据集中随机抽取K歌与target不同的负样本词索引"""
    indices = [None]*K
    for k in range(K):
        newidx =dataset.sampleTokenIdx()
        while newidx ==target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices



def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """基于负采样（Negative Sampling）的 Word2Vec 代价与梯度计算函数。

    对应理论推导 3(c)。

    参数:
    predicted -- 中心词的输入词向量 vc，形状为 (D,)
    target -- 真实期望的上下文目标词在词表中的整型索引 o
    outputVectors -- 所有单词的输出词向量矩阵 U，形状为 (V, D)
    dataset -- 数据集对象，提供 dataset.sampleTokenIdx() 用于随机抽取负样本
    K -- 负样本抽取数量，默认为 10

    返回值:
    cost -- 负采样标量损失 J
    gradPred -- 损失关于预测输入向量 vc 的梯度，形状为 (D,)
    grad -- 损失关于输出词向量矩阵 U 的梯度，形状为 (V, D)
    """
    ### TODO: 在此实现负采样代价与梯度
    # 1. 获取正样本以及 K 个负样本的索引
    indices = [target] + getNegativeSamples(target, dataset, K)
    u = outputVectors[indices]               # (K+1, D)
    scores = np.dot(u, predicted)            # (K+1,)

    # 2. 计算预测概率与交叉熵损失 (对应 3(c) 公式)
    probs = sigmoid(scores)                  # (K+1,)
    # 正样本损失: -log(sigma(u_o^T * v_c))
    # 负样本损失: -sum_{k=1}^K log(sigma(-u_k^T * v_c))
    cost = -np.log(probs[0]) - np.sum(np.log(sigmoid(-scores[1:])))

    # 3. 计算误差项 delta = (预测概率 - 真实标签)
    delta = probs.copy()                     # (K+1,)
    delta[0] -= 1.0                          # 正样本标签为 1，其余负样本标签为 0

    # 4. 向量化计算中心词梯度: sum_i delta_i * u_i -> 形状 (D,)
    gradPred = np.dot(delta, u)

    # 5. 计算并累加输出词向量梯度:
    # 注意: 同一个负样本词可能被重复抽中多次，因此使用 np.add.at 进行安全累加
    grad = np.zeros(outputVectors.shape)     # (V, D)
    np.add.at(grad, indices, np.outer(delta, predicted))

    return cost, gradPred, grad

#word2vecCostAndGradient 是一个“函数参数”（也就是作为参数传进来的函数名）F

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """Skip-gram 模型前向与反向传播。

    对应理论推导 3(d) 公式 (7)。

    参数:
    currentWord -- 当前中心词的字符串（如 "apple"）
    C -- 上下文窗口的半宽大小（即中心词前后各考察 C 个单词）
    contextWords -- 上下文窗口内出现的 2*C 个上下文单词字符串列表
    tokens -- 词典映射字典，将单词字符串映射为整型索引（如 {"apple": 0, ...}）
    inputVectors -- 输入词向量矩阵 V，形状为 (词表大小, D)
    outputVectors -- 输出词向量矩阵 U，形状为 (词表大小, D)
    dataset -- 数据集对象
    word2vecCostAndGradient -- 底层单次预测函数，默认为 softmaxCostAndGradient，
                               也可传入 negSamplingCostAndGradient

    返回值:
    cost -- 当前窗口内所有预测任务的累加总损失
    gradIn -- 损失关于输入词向量矩阵 inputVectors 的累加梯度，形状同 inputVectors
    gradOut -- 损失关于输出词向量矩阵 outputVectors 的累加梯度，形状同 outputVectors
    """
    ### TODO: 在此实现 Skip-gram 上下文滑动窗口循环
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)  #(V,D)
    gradOut = np.zeros(outputVectors.shape) #(V,D)

    c_idx = tokens[currentWord]
    predicted = inputVectors[c_idx]

    for word in contextWords:
        target_idx = tokens[word]
        c,gradPred,grad = word2vecCostAndGradient(predicted,target_idx,outputVectors, dataset)
        cost+=c
        gradOut+=grad
        gradIn[c_idx]+=gradPred

    return cost, gradIn, gradOut



def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW 模型前向与反向传播（选做题 (h)）。

    对应理论推导 3(d) 公式 (8) 与 (9)。
    """
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    target_idx = tokens[currentWord]
    context_indices = [tokens[w] for w in contextWords]
    predicted = np.sum(inputVectors[context_indices], axis=0)
    # 4. 只进行 1 次单步预测 (对应 3(d) 公式 9)
    cost, gradPred, gradOut = word2vecCostAndGradient(
        predicted, target_idx, outputVectors, dataset)
    # 5. 上下文窗口里的每一个词都平分这个梯度 (累加进去)
    for idx in context_indices:
        gradIn[idx] += gradPred

    return cost, gradIn, gradOut


#############################################
# 测试与适配函数（官方提供，用于验证算法）  #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    """将 Word2Vec 模型适配到 SGD 优化器的包装器函数。"""
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N // 2, :]
    outputVectors = wordVectors[N // 2:, :]
    for i in range(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c
        grad[:N // 2, :] += gin
        grad[N // 2:, :] += gout

    cost /= batchsize
    grad /= batchsize

    return cost, grad


def test_normalize_rows():
    """测试矩阵行归一化函数"""
    print("正在测试 normalizeRows...")
    x = np.array([[3.0, 4.0], [1.0, 2.0]])
    # 第一行模长为 sqrt(3^2 + 4^2) = 5.0，归一化后为 [0.6, 0.8]
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    out = normalizeRows(x)
    assert np.allclose(out, ans, rtol=1e-05, atol=1e-06), "normalizeRows 测试未通过"
    print("normalizeRows 测试通过！\n")


def test_word2vec():
    """测试 Word2Vec 模型与梯度的单元测试函数"""
    # 构造一个模拟的虚拟数据集（用于梯度检查）
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
            [tokens[random.randint(0, 4)] for _ in range(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    print("==== 1. 测试 Skip-gram + Softmax ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)

    print("\n==== 2. 测试 Skip-gram + 负采样 (Negative Sampling) ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print("\n==== 3. 测试 CBOW + Softmax ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)

    print("\n==== 4. 测试 CBOW + 负采样 ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
