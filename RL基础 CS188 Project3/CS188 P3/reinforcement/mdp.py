# mdp.py
# ------
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


import random

class MarkovDecisionProcess:

    def getStates(self):
        """
        返回 MDP 中所有状态（格子）的列表。
        返回值类型：list，例如 [(0, 0), (0, 1), ...]
        """
        abstract

    def getStartState(self):
        """
        返回 MDP 的初始状态（起点）。
        """
        abstract

    def getPossibleActions(self, state):
        """
        返回在当前状态 state 下所有合法的动作列表。
        返回值类型：list of str，例如 ['north', 'south', 'east', 'west']
        如果是终点状态，则返回空列表 []。
        """
        abstract

    def getTransitionStatesAndProbs(self, state, action):
        """
        返回由 (nextState, prob) 二元元组组成的列表，
        表示在状态 state 采取动作 action 后，可能转移到的下一状态及其转移概率。
        
        返回值类型：list of (nextState, prob)
        例如：[((0, 2), 0.8), ((1, 1), 0.1), ((0, 1), 0.1)]
        概率之和等于 1.0。
        """
        abstract

    def getReward(self, state, action, nextState):
        """
        获取从状态 state 采取动作 action 并转移到 nextState 时获得的即时奖励 R。
        返回值类型：float，例如 0.0, 1.0 或 -100.0
        """
        abstract

    def isTerminal(self, state):
        """
        如果当前状态是终点状态（Terminal State），则返回 True，否则返回 False。
        按照强化学习惯例，终点状态的未来收益折现期望恒为 0。
        """
        abstract
