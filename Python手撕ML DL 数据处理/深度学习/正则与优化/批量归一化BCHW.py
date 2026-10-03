"""
1388. 批量归一化 BCHW

【题目描述】
在 BCHW 格式的四维张量上按通道（Channel）执行批量归一化（Batch Normalization）：
对每个通道，在 Batch 维以及空间维（Height, Width）上计算均值与方差，
进行标准化后再通过可学习参数 gamma 和 beta 进行仿射变换。

【实现要求】
1. 统计维度为 (B, H, W)，即沿 axis=(0, 2, 3) 统计，每个通道独立计算。
2. 默认平滑项 eps = 1e-5。
3. 算法公式：
   x_hat = (x - mu) / sqrt(var + eps)
   y = gamma * x_hat + beta
4. 伪代码逻辑：
   for each channel c:
       mu, var = mean/var of X[:, c, :, :]
       X[:, c, :, :] = gamma[c] * (X[:, c, :, :] - mu) / sqrt(var + eps) + beta[c]
   return X

【输入与输出】
- 输入：
  - X: 形状为 (B, C, H, W) 的四维 NumPy 数组 / 张量。
  - gamma: 形状为 (C,) 或可通过广播匹配通道维 (1, C, 1, 1) 的数组。
  - beta: 形状为 (C,) 或可通过广播匹配通道维 (1, C, 1, 1) 的数组。
  - eps: float，默认 1e-5。
- 输出：
  - 返回与输入同形的四维张量 (B, C, H, W)。
  - 评测判定：绝对误差 <= 1e-5 或相对误差 <= 1e-5。

【限制条件】
- B, C, H, W <= 16
- gamma, beta 长度等于通道数 C
"""\
#BN
import torch
def batch_normalization(x:torch.Tensor,gamma:torch.Tensor,beta:torch.Tensor,eps:float = 1e-5)->torch.Tensor:
    #1.高精度
    x = torch.as_tensor(x,dtype = torch.float64)
    gamma= torch.as_tensor(gamma, dtype=torch.float64)
    beta= torch.as_tensor(beta, dtype=torch.float64)
    B,C,H,W = x.shape
    #2.断言
    assert x.ndim == 4
    B,C,H,W = x.shape
    assert gamma.numel() == C and beta.numel() ==C #gamma/beta 元素数必须等于通道数C
    #3.参数形状对齐 gamma和beta 拉伸为(1,C,1,1) 我们才好广播相乘再相加
    gamma = gamma.reshape(1,C,1,1)
    beta = beta.reshape(1,C,1,1)
    #4.计算均值和方差 按通道统计 因此要跨越B H W ，dim=(0,2,3)
    #且keepdim = True,算出来的均值形状仍然为四维
    mean = x.mean(dim=(0,2,3),keepdim=True)
    #因为分母 m = B*H*W  避免x.var()默认无偏估计带来的浮点误差
    var = ((x-mean)**2).mean(dim=(0,2,3),keepdim=True)
    #5.标准化和仿射重构
    x_hat = (x-mean)/torch.sqrt(var+eps)
    y = gamma*x_hat +beta
    return y








