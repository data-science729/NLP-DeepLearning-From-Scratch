# Python 面向对象（OOP）重点知识总结（深度学习 & NLP 实战版）

本指南专为深度学习与 NLP 学习者打造。通过结合 PyTorch、Transformer、Dataset 等经典 DL/NLP 代码实例，帮助你迅速掌握 Python 面向对象核心概念。

---

## 目录
1. [类与对象（Class & Object）](#1-类与对象class--object)
2. [构造函数 `__init__` 与 `self` 指针](#2-构造函数-__init__-与-self-指针)
3. [类属性 vs 实例属性](#3-类属性-vs-实例属性)
4. [继承（Inheritance）与 `super()`](#4-继承inheritance与-super)
5. [方法重写（Method Overriding）与多态](#5-方法重写method-overriding与多态)
6. [Python 魔法方法（Dunder Methods）与 PyTorch 机制](#6-python-魔法方法dunder-methods与-pytorch-机制)
7. [封装与私有属性（Encapsulation）](#7-封装与私有属性encapsulation)
8. [总结：深度学习写类思维导图](#8-总结深度学习写类思维导图)

---

## 1. 类与对象（Class & Object）

### 💡 核心概念
* **类（Class）**：模型的**蓝图/设计图**。定义了属性（特征）和方法（行为）。
* **对象（Object / Instance）**：根据设计图造出来的**具体实体/实例**。

### 🤖 NLP / 深度学习对应例子
* **类**：`TokenEmbedding` 类，定义了词向量嵌入层应该怎么接收输入、怎么映射。
* **对象**：`tok_emb = TokenEmbedding(vocab_size=10000, d_model=512)`，根据类造出来的一个具体的神经网络层实体。

```python
# 定义类（设计图）
class SimpleTokenizer:
    pass

# 实例化对象（根据设计图造出具体实体）
tokenizer_a = SimpleTokenizer()
tokenizer_b = SimpleTokenizer()
```

---

## 2. 构造函数 `__init__` 与 `self` 指针

### 💡 核心概念
* **`__init__(self, ...)`**：对象的**初始化方法（构造函数）**。当创建对象（如 `model = MyModel(...)`）时，Python 会**自动触发**这个方法，用于准备和配置该对象所需的参数和子组件。
* **`self`**：代表**当前被创建出来的那个对象本身**。
  * `self.x = x` 的意思就是：“把传入的参数 `x` 绑到**我这个对象**身上，以后在这个对象的其他函数里也能随时用 `self.x` 拿到它”。

### 🤖 NLP / 深度学习对应例子

在 Transformer 的 `TokenEmbedding` 中：

```python
import torch
import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    # __init__ 负责接收超参数，并挂载到 self 上
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model, padding_idx=1)
        
        # 把 d_model 存到 self 身上，方便在 forward 中随时使用
        self.d_model = d_model  

    def forward(self, x):
        # 通过 self.d_model 读取初始化时保存的值
        return super().forward(x) * (self.d_model ** 0.5)
```

> **🔑 笔记诀窍**：只要想在后续的方法（如 `forward`）里使用的变量，都必须在 `__init__` 里通过 `self.变量名 = ...` 存起来！

---

## 3. 类属性 vs 实例属性

### 💡 核心概念
* **实例属性（Instance Attribute）**：绑定在 `self` 上的属性，**每个对象独有一份**。例如不同的模型对象可以有不同的 `d_model`。
* **类属性（Class Attribute）**：定义在类内部、函数体外部的属性，**所有对象共享同一份**。

### 🤖 NLP / 深度学习对应例子

```python
class BertConfig:
    # 类属性：所有 BertConfig 实例共享默认的版本号
    MODEL_TYPE = "bert"

    def __init__(self, hidden_size=768, num_heads=12):
        # 实例属性：每个配置对象根据传入参数各自独立
        self.hidden_size = hidden_size
        self.num_heads = num_heads

config_base = BertConfig(hidden_size=768)
config_large = BertConfig(hidden_size=1024)

print(config_base.hidden_size)   # 输出: 768
print(config_large.hidden_size)  # 输出: 1024
print(config_base.MODEL_TYPE)    # 输出: bert (共享类属性)
```

---

## 4. 继承（Inheritance）与 `super()`

### 💡 核心概念
* **继承**：子类自动拥有父类的所有属性和方法。用 `class ChildClass(ParentClass):` 表示。
  * 避免重复造轮子。例如 PyTorch 已经写好了参数管理、GPU 搬运等复杂逻辑（放在 `nn.Module` 里），我们的自定义模型只需要继承 `nn.Module` 就能直接享有这些功能。
* **`super().__init__()`**：子类调用父类的构造函数，确保父类里定义的各种机制能够正常初始化。

### 🤖 NLP / 深度学习对应例子

```python
# TransformerEmbedding 继承了 PyTorch 官方的 nn.Module
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, device, dropout):
        # 【关键】必须先调用父类 nn.Module 的 __init__，注册 PyTorch 内部机制
        super().__init__()
        
        # 在父类的基础上，组装自己的子模块
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=dropout)
```

---

## 5. 方法重写（Method Overriding）与多态

### 💡 核心概念
* **方法重写**：子类定义了一个与父类同名的方法，**覆盖/替换**父类的默认行为。
* **多态**：不同子类重写了同一个方法，外部用统一的方式调用它们，却能各自产生不同的行为。

### 🤖 NLP / 深度学习对应例子

在 PyTorch 中，`nn.Module` 父类定义了一个空的 `forward` 规范。我们在编写 `TokenEmbedding` 或 `TransformerEmbedding` 时，都在**重写（Override）** `forward` 方法：

```python
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        # ... 初始化位置编码表 ...

    # 重写父类的 forward 方法，指定位置编码前向计算逻辑
    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]
```

无论是 `TokenEmbedding` 还是 `PositionalEmbedding`，虽然它们的内部计算逻辑完全不同，但都重写了 `forward`，所以统一都可以接受输入并返回 Tensor 特征。

---

## 6. Python 魔法方法（Dunder Methods）与 PyTorch 机制

Python 中有一类两边带双下划线的方法，如 `__xxx__`，称为**魔法方法**（Special / Dunder Methods）。它们不需要手动调用，而是在特定语法下**被 Python 自动触发**。

深度学习中最常见的 3 个魔法方法：

| 魔法方法 | 自动触发时机 | 在 DL / NLP 中的经典应用 |
| :--- | :--- | :--- |
| **`__init__`** | 创建对象实例时 `model = Class()` | 神经网络层初始化、准备超参数 |
| **`__call__`** | 把对象当函数调用时 `model(x)` | **PyTorch 核心机制**：触发 Hook 并自动调用 `forward(x)` |
| **`__getitem__`** | 用中括号索引取值时 `dataset[i]` | **PyTorch Dataset 核心**：根据索引提取一条训练样本 |
| **`__len__`** | 用 `len(dataset)` 算长度时 | **PyTorch Dataset 核心**：获取数据集的总样本条数 |

### 示例 1：`__call__` 的秘密（为什么用 `model(x)` 而非 `model.forward(x)`）

```python
class CallableLayer:
    def __call__(self, x):
        print("1. 执行一些前置 Hook 检查...")
        out = self.forward(x)
        print("3. 执行一些后置 Hook 钩子...")
        return out

    def forward(self, x):
        print("2. 执行真正的数据特征计算")
        return x * 2

layer = CallableLayer()
# 像函数一样直接调用对象，会自动触发 __call__
result = layer(5) 
```

### 示例 2：自定义 NLP Dataset 必写的魔法方法

在 NLP 数据处理中，我们经常编写自己的 Dataset：

```python
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    # 魔法方法 1：当调用 len(dataset) 时自动触发
    def __len__(self):
        return len(self.texts)

    # 魔法方法 2：当调用 dataset[idx] 取数据时自动触发
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': self.labels[idx]
        }

# 使用示例
dataset = TextDataset(texts=["I love NLP", "Transformer is great"], labels=[1, 1])
print(len(dataset))       # 自动触发 __len__，输出: 2
print(dataset[0])         # 自动触发 __getitem__，输出第一条样本字典
```

---

## 7. 封装与私有属性（Encapsulation）

### 💡 核心概念
* **封装**：隐藏对象内部的具体实现细节，只暴露必要的对外接口。
* **私有属性/方法**：在变量名或方法名前加 `_`（单下划线，约定俗成的私有）或 `__`（双下划线，强行私有），防止外部随意修改对象内部的敏感状态。

### 🤖 NLP / 深度学习对应例子

在封装注意力机制或位置编码时，内部生成的中间计算矩阵（如位置编码矩阵 `_2i`）通常不需要外部直接改动：

```python
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        # 单下划线变量 _2i：提示外部开发者这是模块内部的私有中间变量，请勿随意修改
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
```

---

## 8. 总结：深度学习写类思维导图

写一个 PyTorch 深度学习模块（如 Transformer 的某一层）时，思考步骤如下：

```text
继承 nn.Module
   │
   ├── 1. def __init__(self, 超参数...):
   │      ├── super().__init__()  <-- 必须第一句调用父类初始化
   │      ├── self.d_model = ...  <-- 挂载传入的配置参数
   │      └── self.sub_module = ... <-- 实例化子层 (Linear, Dropout, Embedding...)
   │
   └── 2. def forward(self, x):   <-- 必须实现前向传播
          ├── tok_emb = self.sub_module(x) <-- 调用子模块计算特征
          └── return out                   <-- 返回最终结果 Tensor
```

---
*文件生成于 Transformer 项目目录：[python面向对象重点知识.md](file:///c:/Users/asus/PycharmProjects/NLP/Transformer/python面向对象重点知识.md)*
