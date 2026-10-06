# qlearningAgents.py
# ------------------
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
from fontTools.misc.bezierTools import epsilon

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning 智能体 (Question 5 & 6)

      需要你实现的函数：
        - computeValueFromQValues    (根据 Q 值计算状态价值 V(s) = max_a Q(s, a))
        - computeActionFromQValues   (根据 Q 值提取最优动作 pi*(s) = argmax_a Q(s, a))
        - getQValue                  (读取 Q(s, a)，若未见过则返回 0.0)
        - getAction                  (epsilon-贪心动作选择策略)
        - update                     (时序差分 TD 核心更新公式)

      你可以直接使用的成员变量(意思就是父类的构造函数已经全部存好，我们不用写了直接用)：
        - self.epsilon   (探索率，epsilon 概率随机探索)
        - self.alpha     (学习率，更新步长)
        - self.discount  (折扣因子 gamma)

      你可以直接调用的成员方法：
        - self.getLegalActions(state) -> 返回当前状态下所有合法的动作列表
    """
    def __init__(self, **args):
        "在此处初始化 Q 值存储结构（推荐使用 util.Counter）..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter() #配一个表


    def getQValue(self, state, action):
        """
          返回 Q(state, action)。
          如果从未见过该 (state, action) 对，应返回 0.0；
          否则返回存储在表中的 Q 值。
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state,action)]
        util.raiseNotDefined()

   #最大值求取
    def computeValueFromQValues(self, state):
        """
          计算状态 state 的价值 V(s)：
          返回 max_action Q(state, action)，即所有合法动作中最大的 Q 值。
          注意：如果没有合法动作（如终点状态），应返回 0.0。
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        max_q = float('-inf')
        for action in actions:
            q = self.getQValue(state,action)
            if q>max_q:
                max_q = q
        return max_q






    def computeActionFromQValues(self, state):
        """
          计算当前状态下的最优策略动作 pi*(s)：
          选择使 Q(state, action) 最大的合法动作。
          注意：
          1. 如果没有合法动作（如终点状态），应返回 None。
          2. 如果存在平局（多个动作的 Q 值相同且最大），应随机打破平局以保证充分探索。
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if not actions:
            return None
        max_q = self.computeValueFromQValues(state)
        #收集所有Q值等于最大值的动作
        best_actions = [a for a in actions if self.getQValue(state,a) == max_q]
        return random.choice(best_actions)





    def getAction(self, state):
        """
          根据 epsilon-贪心策略决定当前状态下采取的动作：
          - 以 self.epsilon 的概率：进行探索，随机从合法动作中选择一个；
          - 以 1 - self.epsilon 的概率：进行利用，选择最优动作（即 computeActionFromQValues）。
          
          注意：如果没有合法动作，直接返回 None。

          提示：
          - 判断是否掷硬币成功可以使用 util.flipCoin(prob)
          - 从列表中随机选一个元素可以使用 random.choice(list)
        """
        # 选择动作
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        if random.random()<self.epsilon:
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)







    #时序差分TD核心更新
    def update(self, state, action, nextState, reward):
        """
          当智能体经历一次环境交互 (state, action, nextState, reward) 时被调用。
          在此处执行 Q 学习的 TD 更新公式：
          sample = reward + gamma * max_a' Q(nextState, a')
          Q(state, action) <- (1 - alpha) * Q(state, action) + alpha * sample

          注意：你无需手动调用此函数，游戏模拟器会自动为你调用。
        """
        "*** YOUR CODE HERE ***"
        sample = reward +self.discount*self.computeValueFromQValues(nextState)
        self.q_values[(state,action)] = (1-self.alpha)*self.getQValue(state,action) +self.alpha*sample




    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "与 QLearningAgent 完全相同，仅在吃豆人游戏中有特定的默认参数"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        这些默认参数可以通过命令行覆盖传参，例如：
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha       - 学习率
        epsilon     - 探索率
        gamma       - 折扣因子
        numTraining - 静默训练轮数（训练结束后停止探索与更新，进入测试阶段）
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # 0 永远代表 Pacman 本身
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        调用 QLearningAgent 的 getAction 方法，并通知底层游戏引擎。
        请勿修改或删除此方法。
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       近似 Q-Learning 智能体 (Question 9)

       你只需要重写两个方法：
       1. getQValue (线性函数近似计算：Q(s, a) = w · f(s, a))
       2. update    (基于特征梯度的权重向量更新)
       所有其他 QLearningAgent 的方法均继承沿用。
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter() # 权重向量 w，Counter 默认初值为 0.0

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          根据特征线性加权计算 Q 值：
          Q(state, action) = w · featureVector = sum_i w_i * f_i(state, action)
          提示：util.Counter 支持直接使用点积操作符或对应键乘积累加。
        """
        "*** YOUR CODE HERE ***"
        #提取当前(state,action)的特征向量f
        features = self.featExtractor.getFeatures(state,action)
        return self.weights*features


    def update(self, state, action, nextState, reward):
        """
          根据一次转移更新特征权重向量 w：
          difference = [reward + gamma * max_a' Q(nextState, a')] - Q(state, action)
          对于每个特征 i:
          w_i <- w_i + alpha * difference * f_i(state, action)
        """
        "*** YOUR CODE HERE ***"
        difference = (reward+self.discount*self.computeValueFromQValues(nextState)-self.getQValue(state,action))
        features = self.featExtractor.getFeatures(state,action)
        for feature in features:
            self.weights[feature] += self.alpha*difference*features[feature]



    def final(self, state):
        "在每局游戏结束时被调用。"
        PacmanQAgent.final(self, state)

        # 训练轮数全部结束时触发
        if self.episodesSoFar == self.numTraining:
            # 可以在此处打印权重 weights 用于调试观察
            "*** YOUR CODE HERE ***"
            pass
