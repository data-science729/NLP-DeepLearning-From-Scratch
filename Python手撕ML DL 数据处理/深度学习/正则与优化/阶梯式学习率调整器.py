"""
题目：阶梯式学习率调度器（StepLR Learning Rate Scheduler）
难度：简单（Easy）
分类：机器学习（Machine Learning）

题目描述：
    编写一个 Python 类 StepLRScheduler，实现基于 StepLR 策略的学习率衰减调度器。
    该类需实现：
    1. __init__(self, initial_lr, step_size, gamma)：
       初始化调度器参数，接收初始学习率 initial_lr（float）、衰减步长 step_size（int）以及乘法衰减因子 gamma（float）。
    2. get_lr(self, epoch)：
       输入当前训练轮次 epoch（int，从 0 开始索引），返回当前 epoch 对应的学习率。
       学习率每隔 step_size 个 epoch 乘以一次 gamma（即衰减一次）。
       返回结果需四舍五入保留 4 位小数（round to 4 decimal places）。
    要求：仅使用 Python 标准库，不依赖第三方库。

核心公式：
    lr_{epoch} = initial_lr * (gamma ** (epoch // step_size))

参数：
    initial_lr (float): 初始基准学习率。
    step_size (int): 学习率衰减的周期轮数（每经历 step_size 个 epoch 衰减一次）。
    gamma (float): 学习率衰减因子（衰减比率，如 0.5、0.1）。

方法：
    get_lr(epoch: int) -> float:
        返回当前 epoch 计算并截断保留 4 位小数后的学习率。

示例：
    输入：
        scheduler = StepLRScheduler(initial_lr=0.1, step_size=5, gamma=0.5)
        print(scheduler.get_lr(epoch=0))
        print(scheduler.get_lr(epoch=4))
        print(scheduler.get_lr(epoch=5))
        print(scheduler.get_lr(epoch=9))
        print(scheduler.get_lr(epoch=10))
    输出：
        0.1
        0.1
        0.05
        0.05
        0.025
    解释：
        - epoch 0~4：衰减次数为 0 // 5 = 0 次，学习率保持 0.1；
        - epoch 5~9：衰减次数为 5 // 5 = 1 次，学习率为 0.1 * 0.5 = 0.05；
        - epoch 10：衰减次数为 10 // 5 = 2 次，学习率为 0.1 * (0.5 ** 2) = 0.025。
"""
import torch

#编写类的__init__函数有两个核心任务
#1.基础断言检验:确保传入的参数在数学和逻辑上合法
#2.保存为实例属性:绑定到self上 供接下来的get_lr随时调用
class StepLRScheduler:
    def __init__(self, initial_lr: float, step_size: int, gamma: float):
        #基础断言检验
        assert initial_lr>0
        assert isinstance(step_size,int) and step_size>0
        assert 0.0<gamma<=1.0
        #存入实例属性
        self.initial_lr = float(initial_lr)
        self.step_size = step_size
        self.gamma = float(gamma)

    def get_lr(self,epoch:int)->float:
        assert isinstance(epoch,int) and epoch>=0
        decay_count = epoch//self.step_size
        lr_epoch = self.initial_lr *(self.gamma**decay_count)
        return round(lr_epoch, 4)


if __name__ == "__main__":
    scheduler = StepLRScheduler(initial_lr=0.1, step_size=5, gamma=0.5)
    print(scheduler.get_lr(epoch=0))
    print(scheduler.get_lr(epoch=4))
    print(scheduler.get_lr(epoch=5))
    print(scheduler.get_lr(epoch=9))
    print(scheduler.get_lr(epoch=10))
