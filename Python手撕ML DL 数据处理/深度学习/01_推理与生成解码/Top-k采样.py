"""
2074. Top-K 采样
难度：中等

【题目描述】
在大语言模型的自回归文本生成中，每一步都会输出整个词表的 logits 得分。
传统的贪心解码、束搜索等最大化式解码方法，会导致生成文本出现重复、语义平淡、陷入循环的 “神经文本退化” 问题。
Top-K 采样是经典的截断式解码策略：仅保留 logits 最高的 K 个候选词元，其余位置的 logits 置为负无穷（对应概率为 0），再对截断后的 logits 计算 Softmax 得到最终采样分布。
该方法通过过滤低概率长尾词元，在生成流畅度与多样性之间取得平衡，是当前大模型推理框架的标准配置之一。

【实现要求】
- 非 Top-K 位置概率为 0；输出保留 4 位小数。
- 并列时取更大下标进入 Top-K（等价于对 logits 升序 argsort 后取末 K 个下标）。

【算法伪代码】
idx = argsort(logits)[-k:]   # ties → larger index
mask others to -inf
return round(softmax(masked), 4)

【输出与判定】
- 返回浮点列表；浮点比较：绝对误差 <= 10^-4 或相对误差 <= 10^-4（满足其一即通过）。

【参数说明】
- logits: 词表 logits
- k: Top-K

【返回值】
- 采样分布

【示例 1】
输入：
logits = [1, 5, 2, 3]
k = 2
输出：
返回值 = [0.0, 0.8808, 0.0, 0.1192]

【限制条件】
- 1 <= k <= vocab (词表大小)
"""
import torch
def top_k_sample(logits:torch.Tensor,k:int)->torch.Tensor:
    logits = torch.as_tensor(logits,dtype = torch.float32)
    #升序排序提取下标 取最后k个 stable=True保证数值并列时 原始下标靠后的排在后面
    idx = torch.argsort(logits,stable=True)[-k:]
    masked = torch.full_like(logits,-float('inf'))
    masked[idx] = logits[idx]
    probs = torch.softmax(masked,dim=-1)
    return torch.round(probs,decimals=4)






