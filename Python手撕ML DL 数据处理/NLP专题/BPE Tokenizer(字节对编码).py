"""
字节对编码（Byte Pair Encoding, BPE）分词器
难度：中等
分类：自然语言处理（NLP）

【题目描述】
实现字节对编码（BPE）算法的训练阶段。BPE 是一种广泛应用于 GPT、RoBERTa 等现代 NLP 模型中的子词（subword）分词方法。
给定一个语料库（表示为一个字典，键为以空格分隔的 token 序列，值为对应的词频）以及需要执行的合并操作次数（num_merges），请实现 BPE 训练算法。

该算法通过迭代方式运行：
1. 统计整个语料库中所有相邻 token 对（token pair）的频次，统计时需乘以对应词的频次（加权统计）。
2. 找出频次最高的一对 token。
3. 在语料库的所有序列中合并该 token 对（将其替换为二者拼接后的新 token）。
4. 记录该次合并操作。
5. 重复上述步骤，直到达到指定的合并次数。

【注意事项】
- 当在一个词中统计 token 对时，若同一 token 对在词中出现多次（例如：'a b a b' 中包含两次 'a b'），
  每次出现均需单独计入，并分别乘以该词的频次。
- 若可合并的 token 对总数少于 num_merges（例如语料库中所有词均已被合并为单个 token，无法再形成配对），则提前终止循环。
- 特殊标记 '</w>' 代表词尾（end-of-word），应作为普通 token 处理，同样参与统计与合并。
- 编写一个函数，返回记录所有合并操作的列表，其中每个合并操作为一个由被合并的两个 token 组成的元组。

【示例】
输入：
corpus = {"h u g </w>": 10, "p u g </w>": 5, "p u g s </w>": 5}, num_merges = 2

输出：
[('u', 'g'), ('ug', '</w>')]

【推导过程】
- 第 1 步：按词频加权统计所有相邻 token 对。
  对 ('u', 'g') 的频次为 10 + 5 + 5 = 20；
  ('h', 'u') 频次为 10；
  ('g', '</w>') 频次为 10 + 5 = 15；
  ('p', 'u') 频次为 5 + 5 = 10；
  ('g', 's') 频次为 5；
  ('s', '</w>') 频次为 5。
  频次最高的是 ('u', 'g')（共 20 次）。在语料库中合并 'u' 和 'g'：
  语料库更新为 {"h ug </w>": 10, "p ug </w>": 5, "p ug s </w>": 5}。

- 第 2 步：重新统计相邻 token 对。
  ('ug', '</w>') 频次为 10 + 5 = 15；
  ('h', 'ug') 频次为 10；
  ('p', 'ug') 频次为 10；
  ('ug', 's') 频次为 5；
  ('s', '</w>') 频次为 5。
  频次最高的是 ('ug', '</w>')（共 15 次）。合并 'ug' 与 '</w>'。

最终记录的合并列表为：[('u', 'g'), ('ug', '</w>')]。
"""
#输入是2个参数:一个是字典(字典里放着单词切分后字符串，结尾带EOW</w>+词频int)，一个是整数(希望合并的最大轮数)
#输出是一个列表，里面放着按照顺序合并的元组(记录每一次合并了哪两个token)
from collections import defaultdict,Counter



def build_bpe_corpus(data: str | dict[str, int], end_of_word: str = '</w>') -> dict[str, int]:
    """
    辅助函数：将原始文本或词频字典转换为 BPE 算法专用的初始输入格式
    """
    # 1. 如果传入的是纯字符串文本，自动统计词频
    if isinstance(data, str):
        word_counts = Counter(data.split())
    else:
        word_counts = data
    # 2. 字符级空格切分 + 拼接词尾标记
    corpus = {}
    for word, freq in word_counts.items():
        # list("low") -> ['l', 'o', 'w']
        # ' '.join(...) -> "l o w"
        # 最终拼接成: "l o w </w>"
        split_word = ' '.join(list(word)) + f' {end_of_word}'
        corpus[split_word] = freq
    return corpus

def bpe(corpus:dict[str,int],num_merges:int)->list[tuple[str,str]]:
    merges = []
    #1.统计当前语料库中所有相邻token pair的频次(乘以词频)
    for _ in range(num_merges):
        pairs = defaultdict(int)
        #.items()字典元组解包，word:接住key freq:接住value
        for word,freq in corpus.items():
            #函数原型:str.split(sep=None,maxsplit=-1)->list[str]
            symbols = word.split()
            #
            for i in range(len(symbols)-1):
                #注意:右边构造元组的语法是 a,b  外面的括号是为了看的更加清晰
                pair = (symbols[i],symbols[i+1])
                pairs[pair]+=freq
        if not pairs:
            break
        #2.找出词频最高的一对token
        #方法dict.get(key,default=None)->Any 找到key 返回该key对应的真实值
        #函数原型 max(iterable,*,key=None,default=None)->Any
        #key:一个接收单个参数的Callable，返回值:输入iterable中的某一个原始元素
        #匿名函数写法:max(pairs,key =lambda k:pairs[k])
        best_pair = max(pairs,key=pairs.get)
        merges.append(best_pair)
        #3.在语料库中合并该pair
        new_corpus = {}
        first,second = best_pair
        for word,freq in corpus.items():
            symbols = word.split()
            #创建一个空列表 用来存放合并后的新token
            new_symbols = []
            i = 0
            #用while循环 能自由控制
            while i < len(symbols):
                if i<len(symbols)-1 and symbols[i] == first and symbols[i+1] ==second:
                    #将当前字符 下一个字符拼起来 然后指针跳过这两个字符 跳到第三个字符去
                    new_symbols.append(first+second)
                    i+=2
                else:
                    #原样保留当前字符 指针只挪动一步 挪到下一个位置继续检查
                    new_symbols.append(symbols[i])
                    i+=1
            #组装成新词并放回语料库 new_symbols: ['h','ug','</w>']
            #new_corpus :"h ug </w>"
            #str.join(iterable:Iterable[str])->str 调用者' '(有空格 作为连接胶水),参数new_symbols,返回拼接好的大字符串

            new_corpus[' '.join(new_symbols)] = freq
        corpus = new_corpus
    return merges


if __name__ == '__main__':
    print("=" * 60)
    print("测试用例：直接使用你笔记流程图里的经典例题！")

    # 1. 原始词频字典（不再需要手动在字母间敲空格和加 </w> 了）
    raw_data = {
        "low": 5,
        "lower": 2,
        "newest": 6,
        "widest": 3
    }

    # 2. 调用辅助函数，自动生成 BPE 专用语料
    # 这里我们传入 end_of_word='_'，完全复刻你流程图里的下划线标记
    corpus = build_bpe_corpus(raw_data, end_of_word='_')

    print("\n【步骤 1】自动构建的 BPE 初始语料：")
    for word, freq in corpus.items():
        print(f"  {word} : {freq}")
    # 3. 运行 6 次合并
    num_merges = 6
    merges = bpe(corpus, num_merges)
    print(f"\n【步骤 2】前 {num_merges} 次合并记录（与你的流程图逐步对照）：")
    for step, (a, b) in enumerate(merges, 1):
        print(f"  第 {step} 次合并: ({a}, {b})  --->  {a + b}")

    print("=" * 60)






