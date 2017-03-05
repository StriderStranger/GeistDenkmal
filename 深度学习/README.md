# 深度学习

## 基础环境

* Ubuntu16.04 + GTX1060 + cuda8.0 + cudnn5.1 + python3.5 + tensorflow0.12

## 基本框架
1. 定义一个映射模型\<MODEL>
 - 训练数据集和标签（TestData，Labels）：placeholder
 - 模型参数：Variable
 - 模型函数：model
 - 模型结果：logits

2. 定义一个损失函数\<LOSS>
 - 表示标签 y_ 和 模型结果 y 的误差距离

3. 选择一个优化算法\<OPTIMIZER>

4. 定义一个生成器\<GENERATOR>
 - 将大量的数据和标签切块（batch），分批送入placeholder中。
 - 处理的是数据的第一维。

5. 初始化并训练\<SESSION.RUN>
 - 在n次循环中，用 optimizer 通过调整 Variable 优化处理 loss。
 - 使 loss 最优的 Variable 与 model 组合成分类器模型。

**关键点在于映射模型（CNN），损失函数（CrossEntropy），优化算法（GradientDescend）的设计与选择**
