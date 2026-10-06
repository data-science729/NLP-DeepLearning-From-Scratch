"""
优先遍历算法 (Prioritized Sweeping Algorithm - PyTorch 版)
难度：困难 (Hard)
领域：强化学习 (Reinforcement Learning)

任务目标：
    使用 PyTorch 张量实现基于模型的强化学习中的优先遍历算法 (Prioritized Sweeping for Model-Based RL)。

背景说明：
    在基于模型的强化学习中，智能体学习环境的模型，并利用模型进行规划更新，无需额外的真实环境交互。
    传统的 Dyna-Q 采用随机采样规划，效率较低；Prioritized Sweeping 通过优先更新那些贝尔曼误差（TD 误差）
    绝对值最大的状态-动作对，从而加速价值函数的收敛。

核心思想：
    维护一个优先队列（记录每个更新的优先级/紧迫度）。当某个状态的 Q 值发生显著更新时，所有能够转移到
    该状态的前驱 (predecessor) 状态-动作对的预期价值也可能受影响，需要计算其新的 TD 误差并推入优先队列。

PyTorch 实现规范：
    1. Q 表：使用 torch.zeros((num_states, num_actions), dtype=torch.float32) 进行初始化。
    2. 环境模型维护：
       记录 (state, action) -> (reward, next_state, done)，可用字典或多维张量暂存。
    3. 前驱状态追踪：
       维护映射关系：next_state -> Set[(s_bar, a_bar)]，表示哪些历史状态动作转移到了该状态。
    4. 经验处理循环：
       遍历真实经验序列 (s, a, r, s_next, done)：
       - 更新模型与前驱集合；
       - 计算当前经验的 TD 误差绝对值：
         target = r if done else r + gamma * torch.max(Q[s_next])
         priority = |target - Q[s, a]|
       - 若 priority > theta，将 (s, a) 插入优先队列（若已存在则保留最大 priority）；
       - 规划循环（最多执行 n_planning 步）：
         * 若队列为空，提前终止本次规划；
         * 弹出优先级最高的状态-动作对 (s_plan, a_plan)；
         * 从已学模型中读取对应的 r_p, s_next_p, done_p；
         * 计算更新目标：target_p = r_p if done_p else r_p + gamma * torch.max(Q[s_next_p])；
         * 更新 Q 表：Q[s_plan, a_plan] = Q[s_plan, a_plan] + alpha * (target_p - Q[s_plan, a_plan])；
         * 遍历以 s_plan 为后继的所有前驱 (s_bar, a_bar)：
             从模型读取前驱的转移预测，计算其优先级 P_bar；
             若 P_bar > theta，将其以 P_bar 插入优先队列（保留更大值）。
    5. 返回值：
       返回更新完毕的 torch.Tensor 类型 Q 表，形状为 (num_states, num_actions)。

参数说明：
    num_states (int): 离散状态数量
    num_actions (int): 离散动作数量
    experiences (list[tuple]): 真实经验序列，元素为 (s, a, r, s_next, done)
        - s (int): 当前状态
        - a (int): 执行动作
        - r (float): 即时奖励
        - s_next (int): 下一状态
        - done (bool): 是否为终止状态
    alpha (float): 学习率
    gamma (float): 折扣因子
    theta (float): 优先队列准入门槛阈值 (priority > theta)
    n_planning (int): 每步真实经验后触发的最大规划步数
    device (torch.device | str, optional): 张量运行设备（默认 'cpu'）

返回值：
    torch.Tensor: 形状为 (num_states, num_actions) 的收敛/规划后 Q 值张量 (dtype=torch.float32)

输入与输出示例：
    输入：
        num_states = 3
        num_actions = 2
        experiences = [
            (0, 0, 0.0, 1, False),
            (1, 1, 1.0, 2, True)
        ]
        alpha = 0.5
        gamma = 0.9
        theta = 0.01
        n_planning = 5
    输出：
        tensor([[0.2250, 0.0000],
                [0.0000, 0.5000],
                [0.0000, 0.0000]], dtype=torch.float32)

推导流程：
    1. 处理 (0, 0, 0.0, 1, False):
       - target = 0.0 + 0.9 * max(Q[1]) = 0.0
       - priority = |0.0 - Q[0, 0]| = 0.0 <= theta，不加入队列，不执行规划。
    2. 处理 (1, 1, 1.0, 2, True):
       - target = 1.0 (done=True)
       - priority = |1.0 - Q[1, 1]| = 1.0 > theta，推入队列。
       - 规划 Step 1:
         弹出 (1, 1)，Q[1, 1] 更新为 0.0 + 0.5 * (1.0 - 0.0) = 0.5。
         前驱为 (0, 0)，对应 target = 0.0 + 0.9 * max(Q[1]) = 0.45。
         前驱 priority = |0.45 - Q[0, 0]| = 0.45 > theta，推入队列。
       - 规划 Step 2:
         弹出 (0, 0)，Q[0, 0] 更新为 0.0 + 0.5 * (0.45 - 0.0) = 0.225。
         状态 0 无前驱，队列为空，本次规划提前结束。
"""