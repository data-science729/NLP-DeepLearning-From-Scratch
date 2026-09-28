"""
================================================================================
                    【ACM 格式题目描述】
题目名称：小红吃冰淇淋之谜 —— HMM 最优隐状态序列解码（维特比算法）
时间限制：1.0s    空间限制：256MB
================================================================================

【题目背景】
你被关在一个没有窗户的密室里，完全无法看到外面的天气情况。
已知每天的天气只有两种可能：热 (Hot) 或 冷 (Cold)，它们构成了系统的隐藏状态序列。
你唯一能够获知的信息，是小红每天在日记本里记录的当天吃冰淇淋的数量（1 个、2 个或 3 个），这构成了观测序列。

现在给定隐马尔可夫模型 (HMM) 的全部参数 λ = (π, A, B)，以及小红连续 T 天吃冰淇淋的记录数据。
请你破译出：在这 T 天里，外界最有可能发生的真实天气序列是什么？
同时输出该最优路径对应的最大联合概率值。

【输入格式 (Input Format)】
- 第 1 行：输入三个正整数 N, M, T，分别表示：
    N: 隐藏状态的种类数（例如天气的种类数，如 Hot、Cold 则 N=2）
    M: 观测值的种类数（例如吃冰淇淋个数种类，1, 2, 3 则 M=3）
    T: 观测序列的长度（天数）
- 第 2 行：输入 N 个以空格分隔的字符串，表示各个隐藏状态的名称（如: Hot Cold）
- 第 3 行：输入 N 个浮点数，表示初始状态概率向量 π (start_prob)，第 i 个数表示第 1 天处于第 i 个状态的概率。
- 接下来的 N 行（第 4 行到第 N+3 行）：每行输入 N 个浮点数，构成 N x N 的状态转移概率矩阵 A (trans_prob)。
    其中第 i 行第 j 个数 A[i][j] 表示前一天状态为 i，今天转移到状态 j 的概率。
- 接下来的 N 行（第 N+4 行到第 2N+3 行）：每行输入 M 个浮点数，构成 N x M 的发射概率矩阵 B (emit_prob)。
    其中第 i 行第 k 个数 B[i][k] 表示处于状态 i 时，产生第 k 种观测值的概率。
- 最后 1 行：输入 T 个整数，表示连续 T 天的观测值序列 O = [o_1, o_2, ..., o_T]（1-indexed，观测值范围在 1 到 M 之间）。

【输出格式 (Output Format)】
- 输出共两行：
  - 第 1 行：输出最可能的隐藏状态序列，各状态名称以空格隔开（例如：Hot Cold Hot）。
  - 第 2 行：输出该最优路径对应的联合概率 P*，四舍五入保留 6 位小数。

--------------------------------------------------------------------------------
【样例输入 (Sample Input)】
2 3 3
Hot Cold
0.6 0.4
0.7 0.3
0.4 0.6
0.1 0.2 0.7
0.8 0.1 0.1
3 1 3

【样例输出 (Sample Output)】
Hot Cold Hot
0.028224

【样例解释】
- 观测序列: 第1天吃3个，第2天吃1个，第3天吃3个。
- 所有可能的天气序列中，路径 "Hot -> Cold -> Hot" 的概率最高：
  P = P(Hot) * P(吃3个|Hot) * P(Cold|Hot) * P(吃1个|Cold) * P(Hot|Cold) * P(吃3个|Hot)
    = 0.6 * 0.7 * 0.3 * 0.8 * 0.4 * 0.7 = 0.028224
- 维特比算法通过动态规划与回溯，精准找到这条最优天气路径！

--------------------------------------------------------------------------------
【数据范围与约定】
- 1 <= N <= 100
- 1 <= M <= 100
- 1 <= T <= 1000
- 保证输入的所有概率值满足 0.0 <= p <= 1.0，且同分布概率之和为 1.0。
- 评分评测时请保证算法时间复杂度不超过 O(T * N^2)。
================================================================================
"""
import sys
import numpy as np

def viterbi_alogorithm(N,M,T,pi,A,B,obs,states):
    delta = np.zeros((T,N),dtype = np.float64)
    psi = np.zeros((T,N),dtype=np.int64)
    first_obs = obs[0]
    delta[0] = pi*B[:,first_obs]
    psi[0] = np.zeros((N,),dtype = np.int64)

    for t in range(1,T):
        curr_obs = obs[t]
        for j in range(N):
            trans_probs = delta[t-1] * A[:,j]
            max_prob = np.max(trans_probs)  #挑出最大值
            best_prev = np.argmax(trans_probs)  #挑出最大值对应的下标(0还是1)
            #填入两张大表 分数乘上今天的发射概率填入delta
            delta[t,j] = max_prob * B[j,curr_obs]
            psi[t,j] = best_prev

    #阶段3:终止 找出最后一天谁得分最高
    best_prob = np.max(delta[-1])
    best_last_state = np.argmax(delta[-1])
    #阶段四:顺藤摸瓜 倒序回溯整条最优路径
    best_path = [0] * T
    best_path[-1] = best_last_state
    for t in range(T-1,0,-1):
        best_path[t-1] = psi[t,best_path[t]]
    best_path_names = [states[i] for i in best_path]
    return best_path_names,best_prob









def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    iterator = iter(input_data)
    N = int(next(iterator))
    M = int(next(iterator))
    T = int(next(iterator))
    states = [next(iterator) for _ in range(N)]
    pi = np.array([float(next(iterator)) for _ in range(N)], dtype=np.float64)
    A = np.array([[float(next(iterator)) for _ in range(N)] for _ in range(N)], dtype=np.float64)
    B = np.array([[float(next(iterator)) for _ in range(M)] for _ in range(N)], dtype=np.float64)
    obs = np.array([int(next(iterator)) - 1 for _ in range(T)], dtype=np.int64)

    best_path_names, best_prob = viterbi_alogorithm(N, M, T, pi, A, B, obs, states)
    print(*best_path_names)
    print(f"{best_prob:.6f}")
if __name__ == "__main__":
    main()



