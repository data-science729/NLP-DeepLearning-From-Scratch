"""
================================================================================
                    【ACM 格式题目描述】
题目名称：小红吃冰淇淋之谜 —— HMM 观测序列概率评估（前向算法）
时间限制：1.0s    空间限制：256MB
================================================================================

【题目背景】
你被关在一个没有窗户的密室里，完全无法看到外面的天气情况。
已知每天的天气只有两种可能：热 (Hot) 或 冷 (Cold)，它们构成了系统的隐藏状态序列。
你唯一能够获知的信息，是小红每天在日记本里记录的当天吃冰淇淋的数量（1 个、2 个或 3 个），这构成了观测序列。

现在给定隐马尔可夫模型 (HMM) 的全部参数 λ = (π, A, B)，小红记录了一段连续 T 天的吃冰淇淋数据。
请你计算并输出：在该模型参数下，观测到该序列出现的全局总概率 P(O | λ)。

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
- 输出一行，包含一个浮点数，表示观测序列出现的总概率 P(O | λ)。
- 结果四舍五入保留 6 位小数。

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
0.040944

【样例解释】
- N=2 (Hot, Cold), M=3 (1, 2, 3 个冰淇淋), T=3 (观测了 3 天)
- 初始概率: P(Hot)=0.6, P(Cold)=0.4
- 转移矩阵 A:
    Hot  -> Hot: 0.7, Hot  -> Cold: 0.3
    Cold -> Hot: 0.4, Cold -> Cold: 0.6
- 发射矩阵 B:
    Hot:  P(1)=0.1, P(2)=0.2, P(3)=0.7
    Cold: P(1)=0.8, P(2)=0.1, P(3)=0.1
- 观测序列: 第1天吃3个，第2天吃1个，第3天吃3个。
- 动态规划前向传播后，第3天终止状态概率之和为 0.040944。

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


def forward_algorithm(N,M,T,pi,A,B,obeservations):
    '''

    :param N: 隐藏状态数 天气数量2
    :param M:观测值种类数 如吃冰淇淋的类别3
    :param T:观测序列的总天数 3
    :param pi:初始状态概率向量
    :param A:状态转移矩阵 N*N  A[i][j]表示从状态i转移到状态j的概率
    :param B:发射概率矩阵 N*M  B[i][k]表示状态i产生观测k的概率
    :param obeservations:观测序列 一维列表[   ]   （T,)
    :return:浮点数 表示观测序列出现的总概率P(O|λ)
    '''
    #阶段一，第0天初始化
    dp = np.zeros((T,N),dtype=np.float64)  #dp[t]是一个长度为N的一维行向量 存着第t天每种天气的概率值
    first_obs = obeservations[0]
    dp[0] = pi * B[:,first_obs]  # :代表所有行 只要第一天观测对应的那一列 形状为(N,)的列向量
    #阶段二，递推推导t=1 到 T-1
    for t in range(1,T):
        curr_obs = obeservations[t]
        dp[t] = (dp[t-1]@A)*B[:,curr_obs]
    #阶段三:终止求和 最后一行的N个天气概率相加 就是走完后的全局总概率
    total_prob = np.sum(dp[-1])  #dp[-1]代表矩阵的最后一行(第T-1天)
    return float(total_prob)




def main():
    input_data = sys.stdin.read().split();  #快速读取所有控制台输入
    if not input_data:
        return

    iterator = iter(input_data)
    N = int(next(iterator))
    M = int(next(iterator))
    T = int(next(iterator))

    # 读取状态名列表并建立映射 (例如 Hot->0, Cold->1) 需要是普通的python列表
    states = [next(iterator) for _ in range(N)]
    # 读取初始状态概率 pi 长度为 N
    pi = np.array([float(next(iterator)) for _ in range(N)], dtype=np.float64)
    # 读取状态转移矩阵 A (N 行 每行 N 个浮点数)
    A = np.array([[float(next(iterator)) for _ in range(N)] for _ in range(N)], dtype=np.float64)
    # 读取发射矩阵概率 B (N 行 每行 M 个浮点数)
    B = np.array([[float(next(iterator)) for _ in range(M)] for _ in range(N)], dtype=np.float64)

    # 读取 T 个观测值序列 转换成 0-indexed 的 0 1 2 方便查矩阵 B
    observations = np.array([int(next(iterator)) - 1 for _ in range(T)], dtype=np.int64)
    # 运行前向传播算法
    ans = forward_algorithm(N, M, T, pi, A, B, observations)

    print(f"{ans:.6f}")   # 保留 6 位小数输出

if __name__ == "__main__":
    main()



