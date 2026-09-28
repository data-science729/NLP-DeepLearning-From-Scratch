# AI & NLP From Scratch: 机器学习、深度学习与核心算子手撕实战

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/NumPy-Vectorized-013243?logo=numpy" alt="NumPy" />
  <img src="https://img.shields.io/badge/Course-Stanford%20CS224N%20%7C%20CS231n-B1040E" alt="Stanford Courses" />
  <img src="https://img.shields.io/badge/Status-Actively%20Developing-success" alt="Status" />
</p>

> **“What I cannot create, I do not understand.” —— Richard Feynman**  
> 本仓库记录了从底层数学推导到工程落地的完整 AI、NLP 与机器学习技术栈。  
> 拒绝浮于表面的调包调参，回归**计算图机制、多元微积分与纯张量向量化（From Scratch）**，构建工业级底层算子实现与严谨的数值对齐测试。

---

## 📌 项目全景知识地图（Repository Structure）

```text
NLP/
├── Python手撕ML DL 数据处理/       # 🔥 核心：从零手写算法与底层算子库
│   ├── 学习笔记markdown/            # 硬核数学推导笔记 (含 Softmax数值防溢出、ResNet导数、LSTM架构推导等)
│   ├── 深度学习/
│   │   ├── 01_推理与生成解码/       # 温度采样、Top-k、Top-p (Nucleus)、Beam Search 束搜索
│   │   ├── 02_损失函数与度量学习/   # 数值稳定 Softmax、KL散度、Hinge Loss 铰链损失
│   │   ├── 03_Transformer与大模型架构/ # 残差分支 Dropout、Attention 权重失活、Pre-Norm 块
│   │   ├── 04_经典算子与优化算法/   # ResNet BasicBlock/Bottleneck、Leaky ReLU
│   │   └── 05_循环神经网络RNN与时序建模/ # 严格对齐 PyTorch 的纯单步 LSTMCell 实现
│   └── 机器学习/
│       └── 01_概率图与序列标注/     # HMM 前向算法、HMM 维特比算法、CRF 维特比解码
│
├── 数据预处理和特征工程/            # 🐼 LeetCode 【30 Days of Pandas】工业级数据处理
│   ├── 01_条件筛选/                # 向量化布尔切片、缺失值过滤
│   ├── 02_字符串函数/              # 正则表达式清洗、字符串格式化
│   ├── 03_数据操作/                # 窗口排名 (Dense Rank)、Nth 高薪水、宽长表透视
│   ├── 04_数据统计/                # 聚合指标统计、条件区间计数
│   ├── 05_数据分组/                # GroupBy、多字段分组与统计聚合
│   └── 06_数据合并/                # 多表关联 (Left/Inner/Cross Join)
│
├── CS224N/                         # 🌲 斯坦福大学顶级自然语言处理课程 (CS224N) 编程实战
│   └── a1programming/              # 词共现矩阵、SVD 降维、Word2Vec 词嵌入实现
│
├── Transformer/                    # 🤖 Transformer 架构进阶实验与注意力机制探究
├── 数值最优化/                      # 📐 凸优化理论、KKT 条件、梯度下降与牛顿法收敛分析
├── NLP课堂/                        # 🎓 核心自然语言处理经典模型与序列标注实验
└── 数据可视化/                      # 📊 统计学特征分布、降维流形与图表呈现
```

---

## 🌟 核心工程特色与自研规范（Highlights）

### 1. 严格践行“零循环纯向量化（Zero-Loop Vectorization）”
* 彻底摆脱原生 Python 慢速 `for` 循环，大量利用张量广播（Broadcasting）、矩阵点积（`@` / `matmul`）与布尔掩码（Mask）；
* 保证算法在大批量（Batch）场景下具备高吞吐、GPU 硬件友好的运算特征。

### 2. 脱离自动求导（Autograd）的纯解析梯度推导
* 对每一个经典算子（如 ResNet 残差块、Softmax 损失、BatchNorm）均完成了白板级的偏导链式推导；
* 深入探究了残差连接中恒等映射 $+I$ 抵御梯度消失的微分几何机理。

### 3. 数值稳定性防御机制（Numerical Stability）
* **防下溢/防上溢**：Softmax 实现中融入 $\max(x)$ 偏移与 Log-Sum-Exp 技巧；
* **除零防御**：在归一化与除法中引入 $\epsilon = 10^{-12}$ 微小量截断。

### 4. 工业级断言与位级（Bitwise）精度校验
* 每一道手撕代码均配备标准 `assert` 测试套件；
* 部分核心模块（如 `手撕LSTM.py`）与 PyTorch 官方原生 `nn.LSTMCell` 进行全权重绑定与位级数值对比，最大绝对误差控制在 $1.1 \times 10^{-16}$ 级别。

---

## 🚀 快速开始与本地运行

```bash
# 1. 克隆本仓库
git clone https://github.com/你的用户名/NLP-and-DeepLearning-From-Scratch.git
cd NLP-and-DeepLearning-From-Scratch

# 2. 创建并激活虚拟环境 (可选)
conda create -n nlp_env python=3.10 -y
conda activate nlp_env

# 3. 安装最小依赖
pip install torch numpy pandas

# 4. 运行任意测试算子 (以 ResNet 残差块为例)
python "Python手撕ML DL 数据处理/深度学习/04_经典算子与优化算法/残差块简单版.py"
```

---

## 📖 学习笔记精选索引

* 📄 [CS231n通关手撕路线图与DeepML题单全景对照.md](Python手撕ML%20DL%20数据处理/学习笔记markdown/CS231n通关手撕路线图与DeepML题单全景对照.md)
* 📄 [Python向量化编程手撕指南.md](Python手撕ML%20DL%20数据处理/学习笔记markdown/Python向量化编程手撕指南.md)
* 📄 [数值稳定的Softmax数学推导.md](Python手撕ML%20DL%20数据处理/学习笔记markdown/数值稳定的Softmax数学推导.md)
* 📄 [LSTM算法学习与结构推导.md](Python手撕ML%20DL%20数据处理/学习笔记markdown/LSTM算法学习与结构推导.md)
* 📄 [Top-p采样算法学习.md](Python手撕ML%20DL%20数据处理/学习笔记markdown/Top-p采样算法学习.md)
* 📄 [束搜索算法学习.md](Python手撕ML%20DL%20数据处理/学习笔记markdown/束搜索算法学习.md)

---

## 📬 个人与项目状态
本项目持续更新中，作为本人备战 **985 院校人工智能方向研究生复试** 与 **一线大厂核心算法岗手撕面试** 的主力代码底座。欢迎交流与交流学习！
