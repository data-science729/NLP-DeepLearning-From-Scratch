# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        *在阅读此文件前，建议先浏览 learningAgents.py*

        ValueIterationAgent（值迭代智能体）：
        在初始化时接收一个马尔可夫决策过程（MDP，见 mdp.py），
        使用给定的折扣因子（discount）运行指定轮数（iterations）的值迭代。
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          你的值迭代智能体应该在构造函数中接收一个 mdp，
          运行指定轮数的值迭代计算（离线规划求解），
          随后根据最终计算出的最优价值表（self.values）行动。

          你可以使用的核心 MDP 方法：
              mdp.getStates()                                -> 获取地图所有状态/格子
              mdp.getPossibleActions(state)                  -> 获取状态下的合法动作列表
              mdp.getTransitionStatesAndProbs(state, action) -> 获取转移后的 (nextState, prob) 列表
              mdp.getReward(state, action, nextState)        -> 获取即时奖励 R
              mdp.isTerminal(state)                          -> 判断是否为终点状态
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # Counter 本质是带默认值 0 的字典（查不存在的 key 返回 0.0）
        self.runValueIteration()

    def runValueIteration(self):
        # 在此处编写值迭代的核心循环代码（Q1）
        # 贝尔曼最优迭代公式：V_k+1(s) = max_a sum_s' T(s,a,s') * [R(s,a,s') + gamma * V_k(s')]
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            new_values = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                best_action = self.computeActionFromValues(state)
                if best_action is not None:
                    new_values[state] = self.computeQValueFromValues(state, best_action)
            self.values = new_values

    def getValue(self, state):
        """
          返回状态 state 的价值 V(s)（已在 __init__ 的值迭代中计算好）。
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          根据存储在 self.values 中的状态价值，计算状态 state 下执行动作 action 的 Q 值。
          公式：Q(s, a) = sum_s' T(s, a, s') * [R(s, a, s') + gamma * V(s')]
        """
        "*** YOUR CODE HERE ***"
        q_value = 0.0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in transitions:
            reward = self.mdp.getReward(state, action, nextState)
            future_value = self.values[nextState]
            q_value += prob * (reward + self.discount * future_value)
        return q_value

    def computeActionFromValues(self, state):
        """
          提取最优策略：根据当前 self.values 中的价值，返回给定状态下的最优动作。
          公式：pi*(s) = argmax_a Q(s, a)

          提示：
          1. 如果没有合法动作（比如终点状态 terminal state），返回 None。
          2. 平局时（多个动作 Q 值相同）可以随意返回其中一个。
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        best_action = None
        max_q = float('-inf')
        for action in actions:
            q = self.computeQValueFromValues(state, action)
            if q > max_q:
                max_q = q
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "返回状态 state 下的最优策略动作（纯利用/贪心，不探索）。"
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        PrioritizedSweepingValueIterationAgent（优先队列值迭代智能体，Q4 加分项）：
        在初始化时接收一个 MDP，使用给定的参数运行优先队列值迭代。
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          初始化函数：接收 mdp、折扣因子、迭代轮数与误差阈值 theta。
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # 在此处编写优先队列值迭代代码（Q4 加分项）
        "*** YOUR CODE HERE ***"

