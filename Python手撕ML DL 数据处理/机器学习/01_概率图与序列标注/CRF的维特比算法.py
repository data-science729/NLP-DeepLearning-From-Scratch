r"""
【题目描述】
给定长度为 N 的序列，状态标签集合大小为 M（状态编号为 1, 2, ..., M）。
已知模型在对数空间下的 N+1 个特征得分矩阵：
  - W_1 (1 × M): 起点 START 到位置 1 各状态的初始得分；
  - W_2, ..., W_N (每个均为 M × M): 位置 i-1 的状态 j 转移到位置 i 的状态 k 的综合特征得分；
  - W_{N+1} (M × 1): 位置 N 各状态转移到终点 STOP 的得分。

状态序列 y = (y_1, ..., y_N) 的总得分定义为：
    Score(y) = W_1[y_1] + \sum_{i=2}^N W_i[y_{i-1}, y_i] + W_{N+1}[y_N]

请求出得分最高的最优状态序列 y* 及对应的最高得分。

【输入格式】
第一行两个整数 N, M (1 <= N <= 1000, 1 <= M <= 100)。
第二行 M 个浮点数，表示 W_1。
接下来 N-1 个矩阵分块，每块 M 行、每行 M 个浮点数，依次表示 W_2 到 W_N。
最后输入 M 个浮点数，表示 W_{N+1}。

【输出格式】
第一行输出最优路径最高得分（保留 4 位小数）。
第二行输出 N 个正整数，表示最优状态序列，以空格隔开。

【样例输入】
3 2
1.0 0.5
0.6 0.2
0.1 0.8
0.7 0.3
0.4 0.9
0.3
0.8

【样例输出】
3.0000
2 2 2

【样例说明】
8 条路径中，路径 (2, 2, 2) 得分为 0.5 + 0.8 + 0.9 + 0.8 = 3.0000，为全局最大值。
"""

import sys


import numpy as np


def crf_viterbi(N, M, w_list):
    """
    CRF 维特比解码算法（NumPy 向量化实现）

    参数:
        N (int): 序列长度
        M(int): 状态标签数量
        w_list (list of np.ndarray): 是一个包含N+1个numpy数组的列表
            - w_list[0]: shape (m,), START 到位置 1 的得分
            - w_list[1..n-1]: shape (m, m), 各步的状态转移综合得分矩阵
            - w_list[n]: shape (m,), 位置 n 到 STOP 的得分

    返回:
        best_score (float): 最大总得分
        best_path (list of int): 最优状态序列 (1-based index)
    """
    # 请在此实现 CRF 维特比算法
    #初始化
    delta = np.zeros((N,M),dtype=np.float64)
    psi = np.zeros((N,M),dtype = np.int64)
    delta[0] = w_list[0]
    psi[0] = 0
    #动态规划向前递推
    for i in range(1,N):
        scores = delta[i-1,:,None] + w_list[i]
        delta[i] = np.max(scores,axis=0)
        psi[i] = np.argmax(scores,axis=0)
    #终点结算
    final_scores = delta[N-1] + w_list[N]
    best_score = float(np.max(final_scores))
    best_last_state = int(np.argmax(final_scores))
    #回溯
    best_path =[0]*N
    best_path[-1] = best_last_state
    for i in range(N-1,0,-1):
        best_path[i-1] = psi[i,best_path[i]]
    #把Python内部的0索引 转换为题目要求的1编号
    best_path = [state+1 for state in best_path]
    return best_score, best_path







def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # TODO: 解析输入数据并调用 crf_viterbi
    iterator = iter(input_data)
    N= int(next(iterator))
    M= int(next(iterator))
    W_start= np.array([(float)(next(iterator)) for _ in range(M)],dtype = np.float64)
    #中间N-1个转移矩阵 w_2 .... w_N （每个shape:(M,M)) 实际要输入(N-1)*M*M 个数字
    W_mats = []
    for _ in range(N-1):
        mat = np.array([[(float)(next(iterator)) for _ in range(M)] for _ in range(M)],dtype=np.float64)
        W_mats.append(mat)

    W_stop= np.array([float(next(iterator)) for _ in range(M)],dtype=np.float64)
    #打包成长度为N+1的列表送入算法
    w_list = [W_start] + W_mats + [W_stop]
    best_score,best_path = crf_viterbi(N,M,w_list)
    print(f"{best_score:.4f}")
    print(*best_path)



if __name__ == "__main__":
    main()
