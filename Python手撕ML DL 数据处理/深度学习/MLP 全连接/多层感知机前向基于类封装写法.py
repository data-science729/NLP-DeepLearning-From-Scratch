import torch
import torch.nn as nn

# 1. 定义网络结构
class TwoLayerMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        # 定义两个 Linear 层
        self.fc1 = nn.Linear(in_features, hidden_features, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_features, out_features, dtype=torch.float64)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        y = self.fc2(h)
        return y

# 2. 编写函数：把题目给定的权重灌入 nn.Linear 中
def mlp_forward_module(X, W1, b1, W2, b2) -> torch.Tensor:
    X = torch.as_tensor(X, dtype=torch.float64)
    W1 = torch.as_tensor(W1, dtype=torch.float64)
    b1 = torch.as_tensor(b1, dtype=torch.float64)
    W2 = torch.as_tensor(W2, dtype=torch.float64)
    b2 = torch.as_tensor(b2, dtype=torch.float64)

    in_dim, hidden_dim = W1.shape
    _, out_dim = W2.shape

    # 实例化模型
    model = TwoLayerMLP(in_dim, hidden_dim, out_dim)

    # 把外部的权重和偏置手动赋给 nn.Linear
    # 注意：nn.Linear 的 weight 内部是 (out, in)，所以要转置 .t()
    with torch.no_grad():
        model.fc1.weight.copy_(W1.t())
        model.fc1.bias.copy_(b1.reshape(-1))
        model.fc2.weight.copy_(W2.t())
        model.fc2.bias.copy_(b2.reshape(-1))

    # 前向计算
    return model(X)