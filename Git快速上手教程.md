# 🚀 Git 零基础快速上手保姆级教程

> **适合人群**：CS/人工智能专业初学者、从没用过 Git 但想半小时内快速掌握的同学。  
> **推荐环境**：Windows 终端（PowerShell 或 Git Bash）/ PyCharm 内置终端。

---

## 目录
1. [五分钟搞懂：Git 的核心心智模型](#一五分钟搞懂git-的核心心智模型)
2. [第一步：首次使用的身份配置（仅需配置一次）](#二第一步首次使用的身份配置仅需配置一次)
3. [第二步：手把手沉浸式实战（跟着一步步敲命令）](#三第二步手把手沉浸式实战跟着一步步敲命令)
4. [第三步：体验“后悔药与时光机”](#四第三步体验后悔药与时光机)
5. [第四步：分支（Branch）—— 平行宇宙与大胆做实验](#五第四步分支branch-平行宇宙与大胆做实验)
6. [第五步：在 PyCharm 中使用纯鼠标操作 Git](#六第五步在-pycharm-中使用纯鼠标操作-git)
7. [第六步：深度学习/NLP 学生的特别忠告（.gitignore）](#七深度学习nlp-学生的特别忠告gitignore)
8. [核心高频命令终极速查表](#八核心高频命令终极速查表)

---

## 一、五分钟搞懂：Git 的核心心智模型

为什么有了复制粘贴，我们还需要 Git？  
因为 Git 是一个专业的**代码时光机与存档管理器**。

为了理解 Git，你只需要记住**四个区域**与这个生活比喻：

```
+------------------+      git add       +------------------+     git commit     +------------------+      git push      +------------------+
|    工作区        | -----------------> |     暂存区       | -----------------> |    本地仓库      | -----------------> |    远程仓库      |
| Working Directoy |                    |  Staging Area    |                    | Local Repository |                    | GitHub / Gitee   |
+------------------+                    +------------------+                    +------------------+                    +------------------+
   (你的工作书桌)                           (装箱打包盒)                          (自家的地下保险柜)                       (云端国家图书馆)
```

1. **工作区（Working Directory）**：你平时在 PyCharm 里直接写代码、看文件的地方，随时修改。
2. **暂存区（Staging Area）**：相当于“快递打包盒”。你改了 10 个文件，但这次只想提交 2 个，就把这 2 个放进盒子里准备封箱。
3. **本地仓库（Local Repository）**：自家的保险柜。一旦你“封箱存档”（Commit），这次的改动就永久记录在历史中了，打雷断网都丢不掉。
4. **远程仓库（Remote Repository）**：如 GitHub、Gitee。将本地的保险柜同步推送到云端备份，供别人查看或多人协同。

---

## 二、第一步：首次使用的身份配置（仅需配置一次）

因为每次存档都需要记录“是谁改了这段代码”，所以第一次使用必须先报上名来。

打开终端（PyCharm 下方的 Terminal），运行以下两行命令（将名字和邮箱改成你自己的）：

```bash
git config --global user.name "YourName"
git config --global user.email "your_email@example.com"
```

*检验配置是否成功*：
```bash
git config --global --list
```
看到输出里有你的 `user.name` 和 `user.email` 即表示配置完成！

---

## 三、第二步：手把手沉浸式实战（跟着一步步敲命令）

现在，让我们在一个独立的新建文件夹里，完整体验一次 Git 的生命周期！

### 1. 新建并进入测试目录
```bash
mkdir git_practice
cd git_practice
```

### 2. 初始化 Git 仓库（赋予它时光机功能）
```bash
git init
```
* **输出提示**：`Initialized empty Git repository in .../.git/`
* **含义**：Git 在这个文件夹里生成了一个隐藏的 `.git` 文件夹（不要手动修改它），说明这个项目已经受 Git 监控了。

### 3. 查看当前状态（整个 Git 中最最重要的命令！）
> 💡 **黄金法则**：任何时候不知道该干什么，或者不知道现在处于什么状态，就敲：
```bash
git status
```
* **当前输出**：`nothing to commit (create/copy files and use "git add" to track)` —— 告诉你现在干干净净，啥也没有。

### 4. 新建一个代码文件
在终端执行（或在文件夹里手动新建一个 `hello.py`）：
```bash
echo "print('Hello, Word2Vec!')" > hello.py
```
再次查看状态：
```bash
git status
```
* **输出提示**（标红）：`Untracked files: hello.py`
* **含义**：工作区里出现了一个新文件，但它还没被放进“打包盒”（暂存区）。

### 5. 将文件放入暂存区（打包）
```bash
git add hello.py
```
> 💡 提示：如果修改了多个文件，可以直接用 `git add .`（后面的点代表“把当前目录所有改动都放进去”）。

再次查看状态：
```bash
git status
```
* **输出提示**（标绿）：`Changes to be committed: new file: hello.py`
* **含义**：文件已经打包进盒子了，正等待封箱。

### 6. 提交存档（锁进保险柜）
```bash
git commit -m "第一次存档：创建了基础的 hello.py 文件"
```
* `-m` 代表 **Message（提交信息）**。**极其重要！** 必须用简短清晰的一句话说明这次修改了什么，方便以后回溯。
* 再次敲 `git status`，会发现提示 `working tree clean`（书桌又干净了，改动已完全被存盘保护）。

### 7. 查看历史时光轴（查看所有存档）
```bash
git log --oneline
```
* **输出示例**：
  ```text
  a1b2c3d 第一次存档：创建了基础的 hello.py 文件
  ```
  前面的 `a1b2c3d` 是每次提交独一无二的 **版本哈希 ID（相当于存档编号）**。

---

## 四、第三步：体验“后悔药与时光机”

如果写错了代码，怎么后悔？

### 场景 1：我改乱了代码，还没提交，想瞬间还原回刚存档的状态
假设你在 `hello.py` 里胡乱加了一行垃圾代码：
```bash
echo "asdkjhfkjasdf" >> hello.py
```
此时你后悔了，不想改了：
```bash
git restore hello.py
```
再次打开 `hello.py`，刚才胡乱写的代码被瞬间抹掉，恢复如初！

---

### 场景 2：比较我到底改了哪些内容（Diff 查看改动）
修改 `hello.py`：
```bash
echo "print('CS224n Assignment 1 is awesome!')" >> hello.py
```
敲击命令：
```bash
git diff
```
* Git 会用绿色显示新增的行，用红色显示删除的行，一目了然！

---

### 场景 3：彻底坐时光机穿梭回过去的某个版本
如果你提交了 10 次，想完全回到第 1 次提交：
```bash
git checkout <存档编号>
# 例如：git checkout a1b2c3d
```
想回到最新的版本：
```bash
git checkout master   # 或 main
```

---

## 五、第四步：分支（Branch）—— 平行宇宙与大胆做实验

在写作业或做科研时，如果你想尝试一个新的模型架构（比如把 Softmax 改成负采样），但又怕改废了影响原来的代码，怎么办？

**答案是：创建一个独立的分支（平行宇宙）！**

```
主分支 (master/main) -------------------------> 最终成品
                         \                  /
新特性分支 (dev)          +--> 试验新功能 --+ (合并进主干)
```

1. **创建并切换到新分支**：
   ```bash
   git checkout -b try_negative_sampling
   ```
2. **在分支上尽情改动和测试并提交**：
   ```bash
   git add .
   git commit -m "实现了负采样算法测试"
   ```
3. **如果实验很成功，切回主分支并把它合并（Merge）进来**：
   ```bash
   git checkout master
   git merge try_negative_sampling
   ```
4. **如果实验失败彻底搞砸了，直接删掉这个分支，主分支毫发无损**：
   ```bash
   git checkout master
   git branch -d try_negative_sampling
   ```

---

## 六、第五步：在 PyCharm 中使用纯鼠标操作 Git

因为你用的是 PyCharm，你甚至大部分时间都不需要敲黑框命令行！

1. **版本控制工具栏**：
   * 在 PyCharm 界面最左侧或最下方，点击 **Git** 标签页，能看到全彩色的历史提交版本树。
2. **可视化改动提示**：
   * 当你修改了某行代码，代码行号右侧会出现**蓝色/绿色条**。
   * 点击那个彩色条，可以直接查看对比、甚至点击“撤销按钮”一键还原那一行。
3. **一键提交（Commit）**：
   * 快捷键 `Ctrl + K`（Windows）：直接弹出图形化提交面板，勾选要提交的文件，输入提交信息，点击 Commit 即可完成！

---

## 七、深度学习/NLP 学生的特别忠告（.gitignore）

做深度学习任务，有一个**致命陷阱**：
> ⚠️ **千万不要把几十 MB、几 GB 的数据集（如 `stanfordSentimentTreebank.zip`）或模型权重（`*.pth`, `*.pkl`, `*.npy`）直接提交进 Git！**  
> 因为 Git 是为追踪文本代码设计的，一旦强行加入巨型二进制文件，Git 仓库会瞬间膨胀卡死，推送到 GitHub 也会直接被拒绝拦截！

### 解决方案：创建 `.gitignore` 文件
在项目根目录下创建一个名为 `.gitignore` 的文件，写入以下规则，告诉 Git 自动无视这些大文件和缓存：

```gitignore
# 忽略 Python 运行时缓存
__pycache__/
*.pyc

# 忽略环境与 IDE 配置
.idea/
.vscode/
venv/

# 忽略数据集与压缩包
*.zip
*.tar.gz
datasets/
stanfordSentimentTreebank/

# 忽略模型检查点与权重
*.npy
*.pkl
*.pth
saved_params_*.npy
```

---

## 八、核心高频命令终极速查表

| 命令 | 通俗含义 | 常用频率 |
| :--- | :--- | :--- |
| `git status` | 查看当前工作区和暂存区状态（红/绿提示） | ⭐⭐⭐⭐⭐（随时敲） |
| `git add .` | 把当前目录下所有修改打包进暂存区 | ⭐⭐⭐⭐⭐ |
| `git commit -m "说明"` | 正式打一个存档点并写入说明 | ⭐⭐⭐⭐⭐ |
| `git log --oneline` | 简短单行查看所有历史存档版本 | ⭐⭐⭐⭐ |
| `git diff` | 查看还没暂存的具体改动细节 | ⭐⭐⭐⭐ |
| `git restore <file>` | 放弃对某个文件的修改（后悔药） | ⭐⭐⭐ |
| `git checkout -b <分支名>`| 创建并切换到一个新分支（平行世界） | ⭐⭐⭐ |
| `git merge <分支名>` | 将分支上的成果合并回主干 | ⭐⭐⭐ |
| `git push origin main` | 将本地代码推送到云端 GitHub | ⭐⭐⭐⭐（需联网配置） |
| `git clone <url>` | 从 GitHub 完整下载克隆别人的开源项目 | ⭐⭐⭐⭐ |

---

> 💡 **练习任务建议**：现在就打开你的终端，用上面的步骤 3 动手建一个 `git_practice` 试一次，5 分钟内你就能彻底掌握 Git 的基础流程！
