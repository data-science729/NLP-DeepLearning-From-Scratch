"""
3438. 残差分支 Dropout
难度: 中等

【题目描述】
归一化能够控制表示的尺度，但子层还需要在不破坏原输入的前提下写入新信息。
残差连接保留一条原样通过的主干，同时只对子层分支做 Dropout，再把两条分支逐元素相加。
实现残差分支合并：对子层输出执行给定的 Dropout，将结果与原输入逐元素相加。
    r = x + Dropout(s)

【实现要求】
- 输入 x 与子层输出 s 必须具有可逐元素相加的相同形状。
- Dropout 只作用于子层输出，残差分支 x 必须原样保留。
- 使用传入的 Dropout 运算，不在函数内创建新的随机失活过程。
- 相加发生在 Dropout 之后。

【输出与判定】
返回残差和，形状与原输入 x 相同。

【参数说明】
- x: 输入张量（如 torch.Tensor 或 numpy.ndarray）
- sublayer_output: 子层输出张量 s，与 x 形状相同
- dropout: 随机失活概率或对应的 Dropout 算子/函数

【返回值】
- 与 x 同形状的张量，表示残差连接后的输出

【示例 1】
输入：
    x (2, 3) = [[1, 1, 1], [1, 1, 1]]
    sublayer_output (2, 3) = [[1, 1, 1], [1, 1, 1]]  # 对应示例隐含的子层输入
    dropout = 0.0
输出：
    返回值 (2, 3) = [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]

【限制条件】
- 最后一维是特征维
- Dropout 遵循 train/eval 模式规范
"""
import math
from typing import Callable,Union
import torch
import torch.nn as nn
def residual_dropout(x:torch.Tensor,sublayer_output:torch.Tensor,dropout:Union[float,Callable])->torch.Tensor:
    x = torch.as_tensor(x,dtype=torch.float64)
    sublayer_output = torch.as_tensor(sublayer_output,dtype=torch.float64)
    #处理dropout算子 (用兼容函数)
    #如果传入的是算子/函数 直接调用它
    if callable(dropout):
        dropped_s = dropout(sublayer_output)
    else:
        p = float (dropout)
        if p ==0.0:
            dropped_s = sublayer_output  #概率为0 不做任何改动
        else:
            dropped_s = torch.nn.functional.dropout(sublayer_output,p=p,training=True)

    r = x +dropped_s
    return r

if __name__ == '__main__':
    print("=" * 60)
    print("测试用例 1: 题目示例 (dropout = 0.0)")
    x1 = [[1, 1, 1], [1, 1, 1]]
    s1 = [[1, 1, 1], [1, 1, 1]]
    res1 = residual_dropout(x1, s1, dropout=0.0)
    print(f"输出:\n{res1}")
    expected1 = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float64)
    assert torch.allclose(res1, expected1), "用例 1 失败！"
    print(">>> 判定: 通过！<<<")
    print("\n" + "=" * 60)
    print("测试用例 2: 传入 nn.Dropout 算子对象")
    drop_layer = nn.Dropout(p=0.0)  # 验证传入算子能否正常运行
    res2 = residual_dropout(x1, s1, dropout=drop_layer)
    assert torch.allclose(res2, expected1), "用例 2 失败！"
    print(">>> 判定: 通过！<<<")
    print("=" * 60)




