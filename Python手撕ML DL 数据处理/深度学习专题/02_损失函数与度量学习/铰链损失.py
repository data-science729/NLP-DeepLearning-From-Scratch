"""
3076. 铰链损失
难度: 简单

【题目描述】
铰链损失（Hinge Loss）是支持向量机（SVM）等“最大间隔（Maximum Margin）”分类模型中最常用的损失函数之一。
它不仅要求样本被正确分类，还要求分类决策具有足够的置信边界（Margin）。
给定 m 个样本的模型原始预测分数数组 scores 和对应的真实类别标签数组 y，请计算并返回所有样本的平均铰链损失。

【数学定义】
对于第 i 个样本，其实际标签为 y_i ∈ {+1, -1}，模型输出的未归一化预测得分为 s_i ∈ R。
单个样本的铰链损失：
    ℓ_i = max(0, 1 - y_i · s_i)
    - 当 y_i · s_i >= 1 时（样本分类正确且位于安全间隔之外），损失为 0；
    - 当 y_i · s_i < 1 时（分类错误或虽然分类正确但落入间隔内），损失为 1 - y_i · s_i。

平均铰链损失：
    L = (1 / m) * Σ_{i=0}^{m-1} max(0, 1 - y_i · s_i)

【输出与判定】
浮点比较：每个数值绝对误差 <= 10^-5 或相对误差 <= 10^-5（满足其一即通过）。

【参数说明】
- scores: List[float]，预测分数数组
- y: List[int]，真实类别标签数组

【返回值】
- float，平均铰链损失值

【示例 1】
输入：
    scores = [0.5, -0.2, 1.2]
    y = [1, -1, 1]
输出：
    返回值 = 0.43333333333333335
解释：
    第 0 个样本：max(0, 1 - 1 * 0.5) = 0.5
    第 1 个样本：max(0, 1 - (-1) * (-0.2)) = max(0, 1 - 0.2) = 0.8
    第 2 个样本：max(0, 1 - 1 * 1.2) = max(0, -0.2) = 0.0
    平均损失 = (0.5 + 0.8 + 0.0) / 3 = 1.3 / 3 ≈ 0.43333333333333335

【限制条件】
- 样本数 1 <= m <= 10^5
- scores 与 y 等长
- 标签 y_i ∈ {+1, -1}
"""
import torch


def hinge_loss(scores: torch.Tensor, y: torch.Tensor) -> float:
    # 转换为 float64 防精度截断
    scores = torch.as_tensor(scores, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)

    # 逐元素向量化计算：max(0, 1 - y * s)
    # y * scores 自动并行逐元素相乘
    losses = torch.relu(1.0 - y * scores)

    # 计算均值并返回 float
    return losses.mean().item()


if __name__ == '__main__':
    print("=" * 60)
    print("测试用例 1: 题目示例")
    scores1 = [0.5, -0.2, 1.2]
    y1 = [1, -1, 1]
    res1 = hinge_loss(scores1, y1)
    print(f"输入: scores = {scores1}, y = {y1}")
    print(f"输出: {res1}")
    print(f"预期: 0.43333333333333335")
    assert abs(res1 - 0.43333333333333335) < 1e-5, "用例 1 失败！"
    print(">>> 判定: 通过！<<<")
    print("=" * 60)



