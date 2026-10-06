# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# 分析与调参题目 (ANALYSIS QUESTIONS) #
######################

# 设置以下参数，使得智能体通过值迭代产生指定的行为策略。

def question2():
    """
    [Question 2] 桥梁穿越分析 (Bridge Crossing Analysis):
    地图是一座宽度为 1 的窄桥，两边都是深渊 (-100)，终点是 (+100)。
    默认情况下 (noise=0.2)，Agent 害怕掉下悬崖不敢过桥。
    请调节 discount 或 noise，使得最优策略敢于跨过桥梁到达右侧 (+100)。
    """
    answerDiscount = 0.9
    answerNoise =0.0
    return answerDiscount, answerNoise

def question3a():
    """
    [Question 3a] 策略偏好：
    1. 偏好近的出口 (+1)；
    2. 愿意冒险走悬崖边（冒险近道）。
    """
    answerDiscount = None
    answerNoise = None
    answerLivingReward = None
    return answerDiscount, answerNoise, answerLivingReward
    # 如果该策略不可能实现，返回 'NOT POSSIBLE'

def question3b():
    """
    [Question 3b] 策略偏好：
    1. 偏好近的出口 (+1)；
    2. 避开悬崖，绕远路走安全通道（稳健近道）。
    """
    answerDiscount = None
    answerNoise = None
    answerLivingReward = None
    return answerDiscount, answerNoise, answerLivingReward
    # 如果该策略不可能实现，返回 'NOT POSSIBLE'

def question3c():
    """
    [Question 3c] 策略偏好：
    1. 偏好远的大奖出口 (+10)；
    2. 愿意冒险走悬崖边（冒险远道）。
    """
    answerDiscount = None
    answerNoise = None
    answerLivingReward = None
    return answerDiscount, answerNoise, answerLivingReward
    # 如果该策略不可能实现，返回 'NOT POSSIBLE'

def question3d():
    """
    [Question 3d] 策略偏好：
    1. 偏好远的大奖出口 (+10)；
    2. 避开悬崖，绕远路走安全通道（稳健远道）。
    """
    answerDiscount = None
    answerNoise = None
    answerLivingReward = None
    return answerDiscount, answerNoise, answerLivingReward
    # 如果该策略不可能实现，返回 'NOT POSSIBLE'

def question3e():
    """
    [Question 3e] 策略偏好：
    永远避开所有出口和悬崖（每一轮游戏永远不结束，活着就有正奖励）。
    """
    answerDiscount = None
    answerNoise = None
    answerLivingReward = None
    return answerDiscount, answerNoise, answerLivingReward
    # 如果该策略不可能实现，返回 'NOT POSSIBLE'

def question7():
    """
    [Question 7] 桥梁穿越重探 (Q-Learning 无模型探索)：
    是否存在一组 (epsilon, learningRate)，使得 Q-learning 在没有任何先验知识的情况下，
    仅训练 50 个 episode 就能以 99% 的概率学会过桥？
    如果存在请填入数值，如果理论上不可能，请返回 'NOT POSSIBLE'。
    """
    answerEpsilon = None
    answerLearningRate = None
    return answerEpsilon, answerLearningRate
    # 如果不可能，返回 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
