# 数值稳定的 Softmax 与 Log-Softmax 数学推导

在机器学习与深度学习手撕面试中，**Softmax 与 Log-Softmax 的数值稳定性**是最高频考查的基础数学功底之一（例如 LeetCode / 大厂机试 1083 题）。本文从计算机底层浮点数表示的缺陷出发，完整推导 **Log-Sum-Exp Trick** 与平移不变性，并给出工业级向量化代码。

---

## 一、为什么朴素计算会“暴毙”？

### 1.1 计算机浮点数上下限（IEEE 754 标准）
计算机表示实数并不是无限精度的，常用精度的数值范围如下：

| 数据类型 | 范围上限 ($\approx$) | 导致上溢的 $x$ (即 $e^x > \text{MAX}$) | 导致下溢为 0 的 $x$ (即 $e^x = 0$) |
| :--- | :--- | :--- | :--- |
| **float32** (单精度) | $3.4 \times 10^{38}$ | **$x > 88.72$** | **$x < -87.33$** |
| **float64** (双精度) | $1.7 \times 10^{308}$ | **$x > 709.78$** | **$x < -708.39$** |

### 1.2 两大数值灾难场景

以题目的数据范围 $|x_i| \le 10^3$ 为例：

#### ① 上溢出（Overflow）引发 `NaN`
设输入向量为 $x = [1000, 1001, 1002]$：
- 直接计算指数：$e^{1000} \to \text{inf}$（无穷大）
- 标准 Softmax：
  $$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{\text{inf}}{\text{inf}} = \mathbf{NaN}$$
- Log-Softmax：
  $$\log(\text{softmax}(x)_i) = x_i - \ln\left(\sum_j e^{x_j}\right) = 1000 - \ln(\text{inf}) = 1000 - \text{inf} = -\text{inf}$$
  若各分量相互减抵，容易出现 $\text{inf} - \text{inf} = \mathbf{NaN}$。

#### ② 下溢出（Underflow）引发除以 0 或 $\ln(0)$
设输入向量为 $x = [-1000, -1001, -1002]$：
- 直接计算指数：$e^{-1000} \to 0$（下溢为零）
- 标准 Softmax 分母求和为 0，出现 $\frac{0}{0} = \mathbf{NaN}$。
- 若先算 Softmax 得到 0，再去取对数：$\ln(0) = -\mathbf{inf}$。

---

## 二、数学推导 1：标准 Softmax 的平移不变性

### 2.1 平移不变性定理
> **定理**：给输入向量的所有分量加上或减去同一个常数 $c$，Softmax 的输出严格不变。

**证明**：
$$
\text{softmax}(x - c)_i = \frac{e^{x_i - c}}{\sum_{j} e^{x_j - c}} = \frac{e^{x_i} \cdot e^{-c}}{\sum_{j} \left(e^{x_j} \cdot e^{-c}\right)} = \frac{e^{x_i} \cdot e^{-c}}{e^{-c} \sum_{j} e^{x_j}} = \frac{e^{x_i}}{\sum_{j} e^{x_j}} = \text{softmax}(x)_i
$$
证毕。

### 2.2 最佳平移量选取
为了让指数运算绝对不上溢，最佳策略是令：
$$m = \max_k(x_k)$$
将原向量 $x$ 替换为 $\tilde{x} = x - m$：
1. **防止上溢**：因为 $m$ 是全向量最大值，所以对任意分量 $j$，必然有：
   $$x_j - m \le 0 \implies e^{x_j - m} \in (0, 1]$$
   最大分量为 $e^0 = 1$。所有指数都在 1 以内，**彻底杜绝上溢**！
2. **防止分母为 0**：因为至少有一个分量取到最大值（即 $x_j = m$），所以：
   $$\sum_j e^{x_j - m} \ge e^{m - m} = e^0 = 1 > 0$$
   分母严格大于等于 1，**绝对不会发生除零错误**！

---

## 三、数学推导 2：Log-Softmax 与 Log-Sum-Exp Trick

在计算损失函数（如 Cross-Entropy）时，我们真正需要的是 $\log(\text{softmax}(x))$。

### 3.1 原始展开
根据对数运算性质：
$$
\log(\text{softmax}(x)_i) = \ln \left( \frac{e^{x_i}}{\sum_{j} e^{x_j}} \right) = \ln(e^{x_i}) - \ln\left( \sum_{j} e^{x_j} \right) = x_i - \ln\left( \sum_{j} e^{x_j} \right)
$$
其中第二项 $\text{LSE}(x) = \ln\left( \sum_{j} e^{x_j} \right)$ 被称为 **Log-Sum-Exp** 函数。

### 3.2 Log-Sum-Exp 的平移消解推导
我们同样引入最大值 $m = \max_k(x_k)$：

1. 将求和项内的指数拆解为：
   $$\sum_{j} e^{x_j} = \sum_{j} e^{(x_j - m) + m} = \sum_{j} \left( e^{x_j - m} \cdot e^m \right)$$
2. 提取公因式 $e^m$ 到求和号外：
   $$\sum_{j} e^{x_j} = e^m \cdot \sum_{j} e^{x_j - m}$$
3. 对两边取自然对数 $\ln$：
   $$
   \ln\left( \sum_{j} e^{x_j} \right) = \ln\left( e^m \cdot \sum_{j} e^{x_j - m} \right) = \ln(e^m) + \ln\left( \sum_{j} e^{x_j - m} \right) = m + \ln\left( \sum_{j} e^{x_j - m} \right)
   $$
4. 将化简后的 $\text{LSE}$ 代回 $\log(\text{softmax}(x)_i)$：
   $$
   \log(\text{softmax}(x)_i) = x_i - \left[ m + \ln\left( \sum_{j} e^{x_j - m} \right) \right]
   $$
5. 整理合并 $(x_i - m)$：
   $$
   \mathbf{\log(\text{softmax}(x)_i) = (x_i - m) - \ln\left( \sum_{j} e^{x_j - m} \right)}
   $$

### 3.3 为什么这个式子两端都绝对稳定？
- **项 1：$(x_i - m)$**
  - 不做指数运算，单纯的线性减法，绝无精度丢失或溢出。
- **项 2：$\ln\left( \sum_{j} e^{x_j - m} \right)$**
  - 指数项 $x_j - m \le 0 \implies e^{x_j - m} \le 1$，求和项不会上溢。
  - 存在最大值自身项 $e^{m - m} = e^0 = 1 \implies \sum \ge 1$。
  - 由于和式 $\ge 1$，对其取对数 $\ln(\dots) \ge 0$，**绝对不可能出现对 0 取对数 ($\ln 0 = -\infty$) 的下溢灾难**。

---

## 四、Python / PyTorch 代码落地对比

### 4.1 错误写法 vs 正确写法

#### ❌ 错误做法 1：直接硬算（遇到大数直接 NaN）
```python
# 致命错误：大数会 exp 溢出
return torch.log(torch.exp(x) / torch.sum(torch.exp(x)))
```

#### ❌ 错误做法 2：先算 stable_softmax，再外层套 log（遇到极小数下溢成 -inf）
```python
# 缺陷：如果某个概率非常接近 0，softmax 结果下溢为 0.0，log(0.0) 变成 -inf
s = stable_softmax(x)
return torch.log(s)
```

#### ✅ 正确做法：直接在 Log-Sum-Exp 空间一体化计算
```python
def log_softmax(scores: torch.Tensor) -> torch.Tensor:
    # 兼容 List 与 Tensor 输入
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)

    # 1. 取最大值 (标量)
    m = torch.max(scores)

    # 2. 减去最大值平移
    shifted = scores - m

    # 3. Log-Sum-Exp 技巧
    lse = torch.log(torch.sum(torch.exp(shifted)))

    # 4. (x - m) - lse
    return shifted - lse
```

---

## 五、进阶：高维多 Batch 向量化实现（工业界手撕规范）

在实际大模型/深度学习中，输入一般是多维的，例如：
- 分类任务：`logits` 形状为 `(batch_size, num_classes)`
- NLP 语言模型：`logits` 形状为 `(batch_size, seq_len, vocab_size)`

必须在最后一个维度 (`dim=-1`) 进行归一化，且**务必添加 `keepdim=True`**：

```python
import torch

def log_softmax_multidim(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    通用多维数值稳定 Log-Softmax
    :param x: 任意维度的 Tensor, e.g. (B, C) 或 (B, T, V)
    :param dim: 进行 softmax 归一化的维度，默认最后一维
    :return: 与输入同形状的对数概率 Tensor
    """
    # 必须加 keepdim=True，否则无法在相应轴上自动广播 (Broadcasting)
    m, _ = torch.max(x, dim=dim, keepdim=True)
    shifted = x - m
    lse = torch.log(torch.sum(torch.exp(shifted), dim=dim, keepdim=True))
    return shifted - lse
```

> **核心避坑点（为什么必须 `keepdim=True`？）**：
> 若 `x.shape = (32, 10)`，不用 `keepdim=True` 时 `m = torch.max(x, dim=-1).values` 的形状会退化成 `(32,)`。
> 在执行 `x - m` 时，广播机制会从末尾对齐：`(32, 10)` 与 `(32)` 维度不匹配直接报错 `RuntimeError: The size of tensor a (10) must match the size of tensor b (32)`！保持 `keepdim=True` 后形状为 `(32, 1)`，广播才能正确执行。

---

## 六、面试考点与延伸思考

### Q1: PyTorch 为什么有 `torch.nn.CrossEntropyLoss`，还要有 `torch.nn.NLLLoss`？
- **公式关系**：
  $$\text{CrossEntropyLoss}(x, y) = \text{NLLLoss}(\text{LogSoftmax}(x), y)$$
- **原因**：如果让用户自己先算 `softmax`，再算 `log`，再算交叉熵，在数值上极不稳定且存在两次精度截断。PyTorch 将 `LogSoftmax` 与 `NLLLoss` 融合成一个底层算子 `CrossEntropyLoss`，直接使用稳定的 Log-Sum-Exp 实现，既快又防止下溢。

### Q2: PyTorch 官方原生函数叫什么？
- `torch.log_softmax(x, dim=-1)` 或 `torch.nn.functional.log_softmax(x, dim=-1)`
- 底层还专门提供了 `torch.logsumexp(x, dim=-1)` 算子。
