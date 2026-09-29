# BPE（Byte Pair Encoding，字节对编码）算法学习笔记与核心语法详解

> **学习背景**：  
> 本篇笔记基于大模型分词器（Tokenizer）经典算法题 **《BPE Tokenizer（字节对编码）》** 的手撕推导过程整理。  
> 区别于前面简单的组件搭积木，BPE 是一道标准的 **NLP 文本处理算法题**。本篇记录 BPE 的历史诞生背景、前两个核心步骤的代码推导，以及在编写过程中沉淀的高频 Python 底层语法。

---

## 一、 为什么大模型（LLM）必须使用 BPE？（诞生背景）

在 2015 年以前，传统的自然语言处理（NLP）在将文本喂给模型前，面临着一个**“非此即彼”的两难困境**：

### 1. 方案一：按“整词”分词（Word-level Tokenization）
* **做法**：以空格或标点切分，把 `"unhappiness"` 作为一个独立的 Token。
* **致命痛点 1（词表爆炸）**：英语常用词数十万，形态变化丰富，词表动辄上百万，模型 Embedding 矩阵显存直接吃不消。
* **致命痛点 2（OOV 未登录词，Out-Of-Vocabulary）**：只要遇到未见过的生僻词、新造词（如 `"ChatGPT-ish"`）或拼写错误，模型只能用 `<UNK>`（Unknown）代替。一段文字如果包含几个 `<UNK>`，整句话的语义就彻底丧失了。

### 2. 方案二：按“单字符”分词（Character-level Tokenization）
* **做法**：拆碎成 26 个英文字母和标点符号，如 `'u', 'n', 'h', 'a', 'p', 'p', 'i', 'n', 'e', 's', 's'`。
* **致命痛点 1（序列急剧变长）**：原本 10 个词的句子变成 100 个字符。Transformer 注意力机制的计算复杂度是 $O(S^2)$，序列变长 10 倍，算力直接爆炸 100 倍！
* **致命痛点 2（单字缺乏独立语义）**：单独一个字母 `'h'` 没有独立表意能力。

---

### 3. BPE 的破局：子词分词（Subword Tokenization）
1994 年，Philip Gage 发明了 BPE 原本用于**数据压缩**。2015 年，Sennrich 等人将其借用到 NLP 中，成为了当今大模型的通用标准（**GPT-2/3/4、LLaMA、RoBERTa、Mistral 均基于此**）：

* **核心直觉**：
  * **高频词不拆开**：如 `"the"`、`"dog"` 出现成千上万次，保持为一个整体；
  * **低频罕见词拆成有意义的子词（Subwords）**：如 `"unhappiness"` $\to$ `"un"` + `"happi"` + `"ness"`；
  * **彻底消灭 OOV**：即使遇到完全没见过的全新词汇，最坏情况也能退化为单个字符进行拼装，**永远不会报错 `<UNK>`**！

---

## 二、 算法目标与函数输入输出规范

### 1. 题目目标
在训练语料库中，通过循环迭代 `num_merges` 次：
1. 找出当前语料中**出现频次最高的相邻两个 Token（Best Pair）**；
2. 将它们**合并（Merge）**成一个全新的 Token；
3. 将该合并操作记录在案；
4. 更新语料库，进入下一轮统计与合并。

### 2. 函数签名（Function Signature）与类型规范

```python
def bpe(corpus: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
```

* **输入参数 1：`corpus: dict[str, int]`**
  * 语料库字典。
  * **Key（`str`）**：以空格分隔的当前单词切分序列，末尾带特殊词尾符 `</w>`（如 `"h u g </w>"`）。
  * **Value（`int`）**：该词在语料中出现的**词频**（如 `10`）。
* **输入参数 2：`num_merges: int`**
  * 最大允许执行的合并次数（如 `2`）。
* **返回值（输出）：`list[tuple[str, str]]`**
  * ⚠️ **注意：返回值不是字典，而是按时间顺序记录每次合并的元组列表**！
  * 示例输出：`[('u', 'g'), ('ug', '</w>')]`。

---

## 三、 第一步剖析：相邻 Pair 的词频加权统计

### 1. 核心代码
```python
pairs = defaultdict(int)
for word, freq in corpus.items():
    symbols = word.split()
    for i in range(len(symbols) - 1):
        pairs[(symbols[i], symbols[i + 1])] += freq
```

### 2. 语法点 1：`defaultdict(int)` —— 优雅消灭 KeyError
* **所属模块**：`from collections import defaultdict`
* **原理**：普通字典 `d[key] += 1` 若 `key` 不存在会触发 `KeyError`。
* **机制**：向 `defaultdict` 传入工厂函数 `int`（无参调用 `int()` 返回 `0`）。当访问不存在的键时，**自动初始化为 `0`**，直接进行 `+= freq`，安全省事。

### 3. 语法点 2：`corpus.items()` —— 字典的双变量解包
* **字典的三种取法**：
  * `d.keys()`：仅遍历键（Key）；
  * `d.values()`：仅遍历值（Value）；
  * `d.items()`：同时返回 `(键, 值)` 构成的元组视图。
* **写法**：`for word, freq in corpus.items():` 利用 Python 的**元组解包（Unpacking）**，单行同时拿到单词字符串和对应的词频整数。

### 4. 语法点 3：`word.split()` 函数原型与参数
* **函数原型**：`str.split(sep=None, maxsplit=-1) -> list[str]`
* **参数行为**：
  * `sep=None`（默认）：按照空白字符（空格、`\t`、`\n`）切分，**自动吃掉连续冗余空格**，不留空白碎项。
  * `maxsplit=-1`（默认）：全部分割到底。
* **效果**：`"h u g </w>".split()` $\to$ `['h', 'u', 'g', '</w>']`。

### 5. 语法点 4：为什么是 `range(len(symbols) - 1)`？（减 1 的原因）
* **滑动窗口原理**：$N$ 个排成一排的元素，**相邻两两配对只能配出 $N - 1$ 对**（植树问题/篱笆原理）。
* **防越界防御**：若不减 1，当 `i` 取到最后一个索引时，访问 `symbols[i + 1]` 会直接触发 `IndexError: list index out of range`（下标越界）。

### 6. 语法点 5：为什么是 `+= freq`（加权累加）？
* 单词 `"h u g </w>"` 出现了 10 次，说明它内部包含的每一个相邻配对（如 `('u', 'g')`）也随之出现了 10 次。
* 跨单词加权累加后，`pairs[('u', 'g')]` 就能准确汇总该字符对在全篇文本中的总出现频次。

### 7. 语法总结：Python 中圆括号 `()` 的全景形态
在本步中，`pair = (symbols[i], symbols[i + 1])` 的右侧使用了圆括号，其全面语法归类如下：

| 语法场景 | 代码示例 | 说明 |
| :--- | :--- | :--- |
| **元组构造（本代码）** | `(a, b)` | 不可变序列，**可以作为字典的 Key**（列表则不行） |
| **单元素元组陷阱** | `(a,)` | **必须带逗号**，否则 `(a)` 只是普通的数学运算括号 |
| **生成器表达式** | `(x for x in data)` | 惰性计算，随用随算，极度节省内存 |
| **改变运算优先级** | `(a + b) * c` | 改变运算顺序 |
| **函数与类调用** | `len(x)` / `Class()` | 执行可调用对象 |
| **优雅多行折行** | `with (open() as f1, ...):` | PEP 8 推荐折行方式，避免反斜杠 `\` |

---

## 四、 第二步剖析：锁定最高频 Pair（`max` 与 `get` 的精妙配合）

### 1. 核心代码
```python
if not pairs:
    break
best_pair = max(pairs, key=pairs.get)
merges.append(best_pair)
```

### 2. 语法点 1：`max()` 函数原型与参数
* **函数原型**：`max(iterable, *, key=None, default=None) -> Any`
* **参数剖析**：
  * `iterable`：传入字典 `pairs` 时，**Python 默认只拿字典的所有的“键”（即待比选的 token 对元组）进行遍历比大小**。
  * `key`：**裁判评分函数（Callable）**。告诉 `max` 不要按候选人自身（字母拼音顺序）比，而是按 `key(候选人)` 算出来的数值比大小。
* **返回值**：返回使 `key` 函数计算结果最大的**那个原始候选键（Key）**，即最高频的元组 `('u', 'g')`。

### 3. 语法点 2：`dict.get()` 函数原型与安全机制
* **函数原型**：`dict.get(key, default=None) -> Any`
* **对比直接索引 `dict[key]`**：
  * `d[key]`：若 key 不存在，**直接崩溃报 `KeyError`**；
  * `d.get(key, default)`：若 key 不存在，**优雅返回 `default`（默认 None）**，绝不崩溃。

### 4. 语法点 3：为什么传 `key=pairs.get` 而不加括号？
* `pairs.get(...)`（带括号）：是执行函数并获取结果；
* `pairs.get`（不带括号）：是**传递函数对象本身作为回调裁判**。
* `max` 内部在打擂台时，会自动拿每一个候选 key 喂给 `pairs.get(key)` 查出词频数值，最后把词频最高的那对元组选出来。

---

## 五、 当前进度复盘与后续步骤预告

```text
【已完成】
  ├── 步骤 1：遍历全语料，加权统计所有相邻 Token 对的频次 (defaultdict)
  └── 步骤 2：利用 max(pairs, key=pairs.get) 精准锁定最高频 Pair

【待完成（下一步）】
  └── 步骤 3：在全语料中执行合并替换（把 'u' 和 'g' 合并为 'ug'，重构语料库）
```

> **阶段思考总结**：  
> 与单纯的数学算子不同，BPE 算法要求对字符串、列表切片、双指针滑动窗口以及字典键值对有极强的掌控力。第一步打通加权统计、第二步打通极值裁判后，最核心的骨架已经完全明朗。
