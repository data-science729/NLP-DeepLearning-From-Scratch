# CS 188 Fall 2026: Project 3 - 强化学习 (Reinforcement Learning) 完整项目指南

> **课程官网**: [UC Berkeley CS 188: Introduction to Artificial Intelligence](https://inst.eecs.berkeley.edu/~cs188/fa26/projects/proj3/#introduction)  
> **项目主题**: 马尔可夫决策过程 (MDP)、值迭代 (Value Iteration) 与 Q 学习 (Q-Learning)  
> **运行环境**: Python 3.x (本项目已在 Python 3.12 环境验证兼容)

---

## 目录 (Table of Contents)

1. [项目简介 (Introduction)](#1-项目简介-introduction)
2. [项目文件架构与代码清单](#2-项目文件架构与代码清单)
3. [快速上手与测试评测命令 (Autograder)](#3-快速上手与测试评测命令-autograder)
4. [核心环境介绍：网格世界 (Gridworld)](#4-核心环境介绍网格世界-gridworld)
5. [项目各题目详析与实现指南](#5-项目各题目详析与实现指南)
   - [Question 1 (5分): 值迭代 (Value Iteration)](#question-1-5分-值迭代-value-iteration)
   - [Question 2 (1分): 桥梁穿越分析 (Bridge Crossing Analysis)](#question-2-1分-桥梁穿越分析-bridge-crossing-analysis)
   - [Question 3 (6分): 策略设计与参数调优 (Policies)](#question-3-6分-策略设计与参数调优-policies)
   - [Question 4 (1分 加分项): 优先队列值迭代 (Prioritized Sweeping VI)](#question-4-1分-加分项-优先队列值迭代-prioritized-sweeping-value-iteration)
   - [Question 5 (5分): Q 学习算法 (Q-Learning)](#question-5-5分-q-学习算法-q-learning)
   - [Question 6 (2分): $\epsilon$-贪心探索与爬行机器人 (Epsilon-Greedy & Crawler)](#question-6-2分-epsilon-贪心探索与爬行机器人-epsilon-greedy--crawler)
   - [Question 7 (1分): 桥梁穿越重探 (Bridge Crossing Revisited)](#question-7-1分-桥梁穿越重探-bridge-crossing-revisited)
   - [Question 8 (1分): 吃豆人 Q 学习 (Q-Learning and Pacman)](#question-8-1分-吃豆人-q-学习-q-learning-and-pacman)
   - [Question 9 (4分): 近似 Q 学习 (Approximate Q-Learning)](#question-9-4分-近似-q-学习-approximate-q-learning)
   - [Q10 (1分): AI 使用反思与合作者声明](#q10-1分-ai-使用反思与合作者声明)
6. [核心避坑指南与高频 Bug 汇总](#6-核心避坑指南与高频-bug-汇总)
7. [知识脉络全景图 (Theory Roadmap)](#7-知识脉络全景图-theory-roadmap)

---

## 1. 项目简介 (Introduction)

在本项目中，你将实现经典强化学习中的两大核心支柱：**值迭代 (Value Iteration)** 与 **Q 学习 (Q-Learning)**。

你将遵循由浅入深、由理论规划到在线试错的实践路径：
1. 首先在课堂标准的**网格世界 (Gridworld)** 环境中测试智能体算法；
2. 随后将算法部署至受物理模拟驱动的**双关节爬行机器人 (Crawler)**，让机器人通过试错学会爬行；
3. 最终应用到经典的**吃豆人 (Pacman)** 游戏中，并借助**近似 Q 学习 (Approximate Q-Learning)** 攻克大规模状态空间下的智能决策。

### 课程前置核心知识点
- **MDPs I**: 马尔可夫决策过程基本定义、转移概率、回报与折扣因子。
- **MDPs II**: 贝尔曼方程 (Bellman Equation)、最优价值函数、值迭代算法。
- **RL I**: 无模型强化学习 (Model-Free RL)、时序差分学习 (Temporal Difference)、Q 学习更新规则。
- **RL II**: 探索与利用权衡 ($\epsilon$-greedy)、特征工程与线性价值函数逼近 (Function Approximation)。

---

## 2. 项目文件架构与代码清单

项目解压后（位于 `reinforcement/` 目录下），各核心文件职责如下：

### 核心修改文件（仅需修改并提交这 3 个文件）
| 文件名 | 作用说明 | 涉及题目 |
| :--- | :--- | :--- |
| `valueIterationAgents.py` | 求解已知 MDP 的值迭代智能体与优先队列值迭代智能体 | Q1, Q4 |
| `qlearningAgents.py` | 用于 Gridworld、Crawler 和 Pacman 的 Q-learning 及近似 Q-learning 智能体 | Q5, Q6, Q9 |
| `analysis.py` | 填入各类情景分析题参数答案与不可行性判断的文件 | Q2, Q3, Q7 |

### 建议研读与调用的支持文件
| 文件名 | 作用说明 | 关键内容 |
| :--- | :--- | :--- |
| `mdp.py` | 定义通用马尔可夫决策过程 (MDP) 的抽象类与接口 | `getStates()`, `getPossibleActions(state)`, `getTransitionStatesAndProbs(state, action)`, `getReward(state, action, nextState)`, `isTerminal(state)` |
| `learningAgents.py` | 定义智能体基类 `ValueEstimationAgent` 和 `QLearningAgent` | 提供了参数管理与继承结构 |
| `util.py` | 常用工具类库 | 提供了专门为强化学习打造的 `util.Counter`（默认值为 0 的字典）、`util.PriorityQueue` 以及 `util.flipCoin(p)` |
| `gridworld.py` | 网格世界模拟器与 MDP 实例 | 包含各种测试迷宫和交互环境 |
| `featureExtractors.py` | 状态特征提取器（用于 Q9） | `IdentityExtractor`（恒等特征）、`SimpleExtractor`（吃豆人专用高级特征） |

### 辅助系统文件（无需修改）
| 文件名 | 作用说明 |
| :--- | :--- |
| `environment.py` | 强化学习环境基类，提供 `getCurrentState()`, `doAction(action)` 等方法 |
| `graphicsGridworldDisplay.py` | 网格世界的图形化渲染界面 |
| `crawler.py` & `graphicsCrawlerDisplay.py` | 爬行机器人的运动学仿真与 GUI 界面 |
| `pacman.py` / `game.py` / `ghostAgents.py` | 吃豆人游戏核心引擎、幽灵行为模型与布局加载器 |
| `autograder.py` / `testParser.py` / `testClasses.py` | 本地自动评分系统与测试框架 |
| `test_cases/` | 包含 Q1 到 Q9 所有评测数据的输入与预期答案目录 |

---

## 3. 快速上手与测试评测命令 (Autograder)

### 3.1 运行自动评分器 (Autograder)
项目自带了非常完善的自动评测脚本，可以在本地直接运行打分：

```bash
# 进入代码目录
cd "RL基础 CS188 Project3/CS188 P3/reinforcement"

# 运行全套测试（评估所有题目）
python autograder.py

# 仅测试指定题目（如测试 Question 1）
python autograder.py -q q1

# 仅测试指定的单个测试用例
python autograder.py -t test_cases/q2/1-bridge-grid
```

### 3.2 界面显示缩放提示
在部分高分屏/非标准分辨率显示器上，GUI 窗口可能会显得非常微小。此时可以通过添加 `--windowSize`（或 `-w`）参数调节大小：
```bash
python gridworld.py -m -w 120
```

---

## 4. 核心环境介绍：网格世界 (Gridworld)

### 4.1 手动控制体验环境
在终端运行以下命令，可以通过方向键手动操控网格世界：
```bash
python gridworld.py -m
```
- **智能体标识**: 界面中的蓝色小圆点即为你操控的 Agent。
- **环境随机性 (Noise)**: 当你按下“向上”时，Agent 只有 **80%** 的概率真正向北移动，各有 **10%** 的概率向东或向西偏转。这就是强化学习中典型的随机动态环境！

### 4.2 随机智能体
观察随机乱走的智能体在迷宫中的表现：
```bash
python gridworld.py -g MazeGrid
```

### 4.3 状态与终止机制细节
- **预终止状态 (Pre-terminal State)**: 界面中显示双边框的方格（如 `+1` 或 `-1`）是预终止状态。
- **终止状态 (Terminal State)**: Agent 必须进入预终止状态后，再执行一个特殊的 `'exit'` 动作，该 episode 才会正式宣告结束（进入 GUI 中不显示的 `TERMINAL_STATE`）。
- **坐标系**: 笛卡尔坐标 `(x, y)`，原点在左下角，`x` 为横轴（向右增大），`y` 为纵轴（向上增大）。`'north'` 对应 y 增加的方向。

### 4.4 常用命令行参数速查
| 参数 | 缩写 | 作用 | 默认值 |
| :--- | :--- | :--- | :--- |
| `--agent` | `-a` | 指定 Agent 类型（`value`, `priosweepvalue`, `q`, `manual`） | `random` |
| `--grid` | `-g` | 指定地图布局（`BookGrid`, `BridgeGrid`, `CliffGrid`, `MazeGrid`, `DiscountGrid` 等） | `BookGrid` |
| `--discount` | `-d` | 折扣因子 $\gamma \in [0, 1]$ | `0.9` |
| `--noise` | `-n` | 动作偏偏离概率（噪声） | `0.2` |
| `--livingReward` | `-r` | 每步存活奖励（负数代表时间惩罚） | `0.0` |
| `--iters` | `-i` | 值迭代执行的轮数 | `10` |
| `--episodes` | `-k` | Q-learning 训练或运行的 episode 回合数 | `1` |
| `--epsilon` | `-e` | $\epsilon$-greedy 探索率 | `0.3` |
| `--learningRate`| `-l` | 学习率 $\alpha$ | `0.5` |
| `--manual` | `-m` | 开启键盘手动操控模式 | 关闭 |
| `--quiet` | `-q` | 静默模式，不打印每一步的转移详情 | 关闭 |

---

## 5. 项目各题目详析与实现指南

---

### Question 1 (5分): 值迭代 (Value Iteration)

#### 理论背景
值迭代是一种**离线规划算法 (Offline Planner)**，假定环境的 MDP 模型完全已知（即转移概率矩阵 $T(s, a, s')$ 和奖励函数 $R(s, a, s')$ 均完全掌握）。

核心贝尔曼更新公式为：
$$V_{k+1}(s) \leftarrow \max_a \sum_{s'} T(s, a, s') \left[ R(s, a, s') + \gamma V_k(s') \right]$$

其对应的 Q 值计算公式为：
$$Q(s, a) = \sum_{s'} T(s, a, s') \left[ R(s, a, s') + \gamma V_k(s') \right]$$

#### 任务要求
在 `valueIterationAgents.py` 中补全 `ValueIterationAgent` 类：
1. `runValueIteration(self)`: 执行指定轮数（`self.iterations`）的更新循环。
2. `computeQValueFromValues(self, state, action)`: 根据当前 `self.values`，计算给定 $(s, a)$ 的 Q 值。
3. `computeActionFromValues(self, state)`: 根据当前 `self.values`，选出在状态 $s$ 下最优的动作 $\pi^*(s) = \arg\max_a Q(s, a)$。

#### 关键实现细节与避坑点
> [!IMPORTANT]
> **必须使用“批处理 (Batch)”模式，严禁就地 (In-Place) 修改！**  
> 在第 $k$ 次迭代计算所有状态的新价值 $V_{k+1}$ 时，右侧所用到的所有后续状态价值必须严格来自于上一轮已经固定的 $V_k$。  
> 做法：在每一轮迭代开始前，克隆一份当前的 `self.values`，在临时字典中收集所有新值，整轮计算完毕后再一次性赋给 `self.values`。

- **终止状态判断**: 如果 `self.mdp.isTerminal(state)` 为 `True` 或当前状态下没有合法动作（`self.mdp.getPossibleActions(state)` 为空），其最优动作为 `None`，其价值不应进行状态转移更新。
- **关于 `util.Counter`**: `util.Counter` 是默认值为 0 的字典。如果所有合法动作的 Q 值均为负数，直接调用 `counter.argMax()` 可能会错误地返回一个未被赋过值的键（因为默认值为 0，大于负数）。务必仅在**当前状态合法的动作集合**中寻找最大 Q 值！

#### 验证命令
```bash
# 运行自动打分
python autograder.py -q q1

# 观察运行 100 轮值迭代后的策略与价值分布
python gridworld.py -a value -i 100 -k 10

# 5 轮迭代的网格图形验证（可对照官网图例）
python gridworld.py -a value -i 5
```

---

### Question 2 (1分): 桥梁穿越分析 (Bridge Crossing Analysis)

#### 题目背景
在 `BridgeGrid` 地图中，左侧是一个较低奖励的出口 (+1)，右侧是一条狭窄悬崖桥对面的极高奖励出口 (+10)。桥的两侧是悬崖（-100 的巨大惩罚）。  
在默认参数（`discount = 0.9, noise = 0.2`）下，因为有 20% 的侧滑概率跌入悬崖，Agent 出于避险心理会选择走向左侧的 +1 出口，而不愿冒险过桥。

#### 任务要求
修改 `analysis.py` 中的 `question2()` 函数：
- **仅修改 `discount` 或 `noise` 中的一个参数**（另一个必须保持默认值不变）；
- 使得在修改后，最优策略敢于跨越独木桥走向 +10 终点。

#### 解题思路
- 为什么 Agent 害怕过桥？因为 `noise = 0.2` 意味着在过桥的每一步都有概率向左或向右跌落深渊。
- 如果将 `noise` 降低到极低值（例如 `0.0` 或 `0.01`），过桥将不再有侧滑跌落的风险，追求高收益的 Agent 自然会勇往直前！
- 或者，通过极大调整 `discount` 使得长远未来的 +10 收益吸引力发生改变（思考哪种更直接有效）。

#### 验证命令
```bash
python autograder.py -q q2
```

---

### Question 3 (6分): 策略设计与参数调优 (Policies)

#### 题目背景
考虑 `DiscountGrid` 布局：
- 中间行包含两个正收益终点：较近终点 (+1) 与较远终点 (+10)；
- 最底下一行是整排悬崖深渊 (-10)；
- 起点位于黄色方块。

#### 任务要求
在 `analysis.py` 中为 `question3a()` 到 `question3e()` 分别配置 `(discount, noise, livingReward)` 三元组，诱导 Agent 产生以下 5 种不同风格的最优策略：

1. **3a: 偏好较近出口 (+1)，且愿意冒跳崖风险 (Risk the cliff)**
   - 短视（低折扣）、不怕微小风险或时间惩罚大。
2. **3b: 偏好较近出口 (+1)，但谨慎绕开悬崖 (Avoid the cliff)**
   - 偏好就近，但噪声较高导致必须远离危险边缘，走安全通道。
3. **3c: 偏好较远出口 (+10)，且愿意冒跳崖风险 (Risk the cliff)**
   - 远见（高折扣），追求最大终极奖励，走距离最短的贴悬崖路线。
4. **3d: 偏好较远出口 (+10)，但谨慎绕开悬崖 (Avoid the cliff)**
   - 远见（高折扣），追求高奖励，但由于环境噪声大，宁可绕远路走上方安全边缘。
5. **3e: 避开所有出口和悬崖 (永不终止)**
   - 让 Agent 觉得活着是一件非常幸福的事（给予正向存活奖励 `livingReward > 0`），这样 Agent 会不断徘徊、拒绝进入任何退出格。

> 如果某种行为在理论上对于任意参数都无法实现，则返回字符串 `'NOT POSSIBLE'`。

#### 验证命令
```bash
# 启动 GUI 调试特定参数
python gridworld.py -g DiscountGrid -a value --discount 0.9 --noise 0.2 --livingReward 0.0

# 自动评分检验
python autograder.py -q q3
```

---

### Question 4 (1分 加分项): 优先队列值迭代 (Prioritized Sweeping Value Iteration)

#### 算法动机
标准值迭代每轮无论状态是否收敛都遍历更新全部状态，存在大量算力浪费。优先队列值迭代根据 **贝尔曼误差 (Bellman Error)** 的大小动态维护一个优先队列，优先更新那些当前价值与真实 Q 值偏差最大的状态，并反向传播给其所有前驱节点。

#### 核心定义
- **前驱节点 (Predecessor)**: 如果从状态 $p$ 执行某个动作 $a$，能以**非零概率**转移到状态 $s$，则 $p$ 是 $s$ 的前驱节点。
- **误差容忍度 $\theta$ (`theta`)**: 只有当潜在更新幅度大于 $\theta$ 时才将其入队更新。

#### 算法流程规范（必须严格遵守）
1. **预处理所有状态的前驱节点**:
   - 遍历所有状态和所有合法动作，构建每个状态的前驱集合（**必须使用集合 `set` 存储以去重**）。
2. **初始化优先队列**:
   - 创建 `util.PriorityQueue` 实例。
   - 按照 `self.mdp.getStates()` 的顺序遍历所有**非终止状态** $s$：
     - 计算 $diff = |self.values[s] - \max_a Q(s, a)|$；
     - 将 $s$ 压入队列，优先级为 **$-diff$**（因为 `PriorityQueue` 是最小堆，负数使最大误差排在最前）。注意：此步**不要**更新 `self.values[s]`。
3. **迭代循环 (`0` 到 `self.iterations - 1`)**:
   - 若优先队列已空，提前终止；
   - 弹出最高优先级（即误差最大）的状态 $s$；
   - 若 $s$ 非终止状态，更新其价值：$self.values[s] \leftarrow \max_a Q(s, a)$；
   - 遍历 $s$ 的每一个前驱节点 $p$：
     - 若 $p$ 非终止状态，计算其当前误差：$diff = |self.values[p] - \max_a Q(p, a)|$；
     - 若 $diff > \theta$，调用 `priorityQueue.update(p, -diff)` 将其推入队列或更新其优先级。

#### 验证命令
```bash
python autograder.py -q q4
python gridworld.py -a priosweepvalue -i 1000
```

---

### Question 5 (5分): Q 学习算法 (Q-Learning)

#### 核心理论
值迭代是**基于模型 (Model-based)** 的规划算法，而 Q 学习是**无模型 (Model-free)** 的时序差分强化学习算法。Agent 在不知道物理世界的转移规律 $T$ 和奖励分布 $R$ 的前提下，直接通过环境试错样本 $(s, a, s', r)$ 来更新动作价值函数：

$$Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') \right]$$

#### 任务要求
在 `qlearningAgents.py` 中补全 `QLearningAgent`：
1. `getQValue(self, state, action)`: 返回 $Q(s, a)$，若未访问过则默认为 $0.0$。
2. `computeValueFromQValues(self, state)`: 计算 $V(s) = \max_a Q(s, a)$。若无合法动作，返回 $0.0$。
3. `computeActionFromQValues(self, state)`: 计算最优动作 $\arg\max_a Q(s, a)$。若无合法动作，返回 `None`。若存在多个相同最大 Q 值的动作，**必须使用 `random.choice` 随机打破平局 (Tie-breaking)**。
4. `update(self, state, action, nextState, reward)`: 根据上式执行单步样本的时序差分更新。

> [!WARNING]
> **抽象隔离原则**:  
> 在 `computeValueFromQValues` 和 `computeActionFromQValues` 中，**只能通过调用 `self.getQValue(state, action)` 来获取 Q 值**，严禁直接读取底层数据字典。这是为了在 Q9 中顺利继承并重写特征 Q 值！

#### 验证命令
```bash
# 手动控制 5 局，观察 Q 学习在足迹中留下的更新
python gridworld.py -a q -k 5 -m

# 自动评分
python autograder.py -q q5
```

---

### Question 6 (2分): $\epsilon$-贪心探索与爬行机器人 (Epsilon-Greedy & Crawler)

#### 理论机制
纯贪心策略会导致智能体陷入局部最优，无法发现潜在的更优路径。$\epsilon$-贪心机制通过一个微小的概率 $\epsilon$ 进行探索：
- 以概率 $\epsilon$：从所有**合法动作**中均匀随机选取一个动作（探索 Exploration）；
- 以概率 $1 - \epsilon$：选择当前 Q 值最大的动作 `self.computeActionFromQValues(state)`（利用 Exploitation）。

#### 任务要求
在 `qlearningAgents.py` 中实现 `QLearningAgent.getAction(self, state)`：
- 使用 `util.flipCoin(self.epsilon)` 判断是否进行探索；
- 探索时使用 `random.choice(legalActions)`；
- 确保没有合法动作时返回 `None`。

#### 爬行机器人实战
完成本题后，你的 Q 学习算法不仅能玩网格游戏，还能直接驱动仿真物理机器人！
```bash
# 启动爬行机器人 GUI
python crawler.py
```
你可以在图形界面中拖动滑块调节学习率 $\alpha$、探索率 $\epsilon$ 以及步长延迟，亲眼见证机器人从胡乱蹬腿到逐步学会协调双臂规律前行的强化学习过程！

#### 验证命令
```bash
python gridworld.py -a q -k 100
python autograder.py -q q6
```

---

### Question 7 (1分): 桥梁穿越重探 (Bridge Crossing Revisited)

#### 思考题
在无噪声的悬崖桥地图（`python gridworld.py -a q -k 50 -n 0 -g BridgeGrid -e 1`）中，Agent 进行 50 回合纯随机学习。  
问题探讨：**是否存在任意一组 $(\epsilon, \alpha)$ 参数，使得 Agent 在短短 50 回合的训练后，有极高概率（>99%）学到跨越桥梁的最优策略？**

在 `analysis.py` 的 `question8()` 中返回答案：
- 若存在，返回元组 `(epsilon, learningRate)`；
- 若不存在，返回 `'NOT POSSIBLE'`。

#### 核心洞察
从桥的起点走到终点需要连续多次做出正确的“向东”动作，而在桥上任何一次向北或向南走就会跌落悬崖导致回合立即结束。在无先验知识且仅有 50 回合的极短时间内，纯靠随机探索成功跨越窄桥且将高收益反向传递到起点的概率在数学上极其微小，因此不可能以 >99% 的确定性学到最优策略。

#### 验证命令
```bash
python autograder.py -q q7
```

---

### Question 8 (1分): 吃豆人 Q 学习 (Q-Learning and Pacman)

#### 任务描述
将之前写好的 `QLearningAgent` 应用于吃豆人环境：
- 吃豆人训练由两个阶段组成：
  1. **训练阶段 (Training)**: 关闭图形界面快速试错，`epsilon` 与 `alpha` 保持正常；
  2. **测试阶段 (Testing)**: 自动将 `epsilon` 与 `alpha` 置为 0，关闭探索并停止学习，纯粹利用学得的最优策略。

#### 运行与评测要求
运行 2000 局训练并在后续 100 局测试中保持 **80% 以上胜率**：
```bash
# 观察最后 10 局测试效果
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

# 自动评分
python autograder.py -q q8
```

#### 传统表格型 Q-Learning 的致命局限
尝试在稍微大一点的地图上运行表格型吃豆人：
```bash
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l mediumGrid
```
你会发现吃豆人几乎必败无疑！  
**原因剖析**: 每一个状态由吃豆人位置、所有幽灵位置及所有豆子分布共同决定。稍微大一点的地图，状态总数呈天文数字级爆炸。表格型方法无法将在状态 A 学到的“靠近幽灵会死”的常识**泛化 (Generalize)** 到形状相似的状态 B。

---

### Question 9 (4分): 近似 Q 学习 (Approximate Q-Learning)

#### 核心原理：价值函数逼近 (Function Approximation)
为了解决维度灾难与泛化问题，我们不再用巨大的表格存储每个状态的独立数值，而是为每个状态-动作对抽取一组特征 $f_1(s, a), \dots, f_n(s, a)$，并维护一组紧凑的特征权重向量 $\mathbf{w}$。

此时 Q 函数被近似为一个线性组合：
$$Q(s, a) = \sum_{i=1}^n f_i(s, a) w_i$$

权重的梯度下降更新公式为：
$$w_i \leftarrow w_i + \alpha \cdot \text{difference} \cdot f_i(s, a)$$
其中时序差分误差项为：
$$\text{difference} = \left[ r + \gamma \max_{a'} Q(s', a') \right] - Q(s, a)$$

#### 任务要求
在 `qlearningAgents.py` 中实现 `ApproximateQAgent`（继承自 `PacmanQAgent`）：
1. 在构造函数中初始化权重字典 `self.weights = util.Counter()`；
2. 重写 `getQValue(self, state, action)`: 计算点积 $\sum f_i \cdot w_i$；
3. 重写 `update(self, state, action, nextState, reward)`: 根据上述公式更新每个特征的权重；
4. 实现 `final(self, state)`: 回合结束时调用父类 `final`，可以在此处做训练轮数日志。

#### 强大威力验证
借助 `featureExtractors.py` 中提取的简单特征（如与最近幽灵的距离、与最近豆子的距离等），仅需 **50 局** 训练即可在中型地图上轻松虐杀幽灵：
```bash
# 中型地图 50 局快速通关
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid

# 经典大地图通关
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic

# 自动打分
python autograder.py -q q9
```

---

### Q10 (1分): AI 使用反思与合作者声明

按照课程要求，在在线问卷或报告中如实填写对生成式 AI 工具（如 LLM、代码助手）的辅助使用记录与队友合作情况。

---

## 6. 核心避坑指南与高频 Bug 汇总

| 序号 | 常见易错陷阱 | 典型症状 / 报错 | 避坑应对建议 |
| :---: | :--- | :--- | :--- |
| **1** | **`util.Counter` 默认值陷阱** | 全负回报状态下选错动作 | 若未见过的动作或特征 Q 值为 0，而当前所有动作 Q 值皆为负，`counter.argMax()` 会误返回 0 对应的伪动作。**一定要限制在 `getLegalActions` 列表内取最大值**。 |
| **2** | **值迭代就地修改 (In-Place)** | Q1 测试出现值偏差或收敛过快 | 每轮更新前必须浅拷贝一份旧 values：`oldValues = self.values.copy()`，计算右侧 Bellman 公式时全部从 `oldValues` 读取，算完一整轮再统一赋值给 `self.values`。 |
| **3** | **平局破决 (Tie-Breaking)** | Q5 测试用例无法完全匹配基准答案 | 当有多个合法动作具有完全相同的最优 Q 值时，不能只取首个，必须用 `random.choice(bestActions)` 随机选取。 |
| **4** | **破坏面向对象多态性** | Q9 特征权重更新了，但吃豆人依旧像没学过一样乱走 | 检查 `computeActionFromQValues` 和 `computeValueFromQValues`，必须统一调用 `self.getQValue(state, action)`，不能自行读取内部属性。 |
| **5** | **优先队列未加负号** | Q4 优先队列值迭代死循环或顺序颠倒 | `util.PriorityQueue` 是 Min-Heap（越小越先出队）。因为我们要优先处理**误差最大**的状态，所以入队优先级必须传入 **`-diff`**。 |
| **6** | **终止状态后续价值不为 0** | 终止节点的 Q 值计算产生幽灵后续回报 | 若 `self.mdp.isTerminal(nextState)`，则该后续状态未来价值必须严格为 0，不再参与折扣累加。 |

---

## 7. 知识脉络全景图 (Theory Roadmap)

本项目精准还原了强化学习从经典控制论走向现代 AI 的技术演化历程：

```
[环境模型已知 (Model-based)]
       │
       ▼
【马尔可夫决策过程 (MDP)】
       │
       ├──> 贝尔曼最优方程 (Bellman Optimality Eq.)
       └──> 动态规划规划器:
             ├── [Q1] 标准值迭代 (Batch Value Iteration)
             └── [Q4] 优先队列值迭代 (Prioritized Sweeping)
       
[环境模型未知 (Model-free)]
       │
       ▼
【无模型强化学习 (Temporal Difference)】
       │
       ├──> 蒙特卡洛 / TD 时序差分误差: δ = r + γ max Q(s',a') - Q(s,a)
       └──> 在线试错学习:
             ├── [Q5] 表格型 Q-Learning (Tabular Q-Learning)
             └── [Q6] 探索与利用权衡 (ε-Greedy Exploration)
       
[状态空间维度灾难 (Curse of Dimensionality)]
       │
       ▼
【函数逼近 (Function Approximation)】
       │
       └──> [Q9] 线性近似 Q 学习 (Approximate Q-Learning with Linear Features)
             └── (注: 深度强化学习 DQN 即是将此处的线性特征替换为深度神经网络)
```

祝你在 CS 188 Project 3 的强化学习探索旅程中顺利通关！全满分在向你招手！
