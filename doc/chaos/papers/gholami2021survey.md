# A Survey of Quantization Methods for Efficient Neural Network Inference

{cite}`gholami2021survey`

量化问题
:   一组连续实数以何种方式分布在一组固定的离散数集上，以最小化所需的比特数，同时最大化 **伴随计算** （attendant computation）的 accuracy。

虽然 CNN 的 over-parameterized 可以显著提高 accuracy，但受制于硬件。普适深度学习需要在资源受限的情况下实现：

- 实时推理
- 低能耗
- 高 accuracy

## 不同的设计方法

### 设计高效 NN 架构

micro-architecture
微架构
:   kernel types，如深度卷积（depth-wise convolution）或 低秩分解（low-rank factorization）

macro-architecture
宏架构
:   module types，如 residual 或 inception

利用 AutoML（Automated machine learning）或者 NAS（Neural Architecture Search）寻找新的架构：旨在以自动化的方式，在给到模型大小、深度和/或宽度的约束下，找到更好的神经网络架构。

### NN 架构与硬件的协同设计

架构与硬件的初始化状态是手动设计的，而 adapt/change NN 架构使用自动化方法。

### 剪枝

剪枝（pruning）目标：通过去除对模型输出或者损失函数影响最小的神经元，以减少 NN 内存占用和计算成本。

剪除具有小显著性（saliency）的神经元（sensitivity），获得稀疏的计算图。

## 量化简介

量化
:   将大集合（通常是连续的）中的值映射到小集合（通常是有限的）。典型的策略有：Rounding（舍入）和截断（truncation）。香农（Shannon）提出 可变速率量化（variable-rate quantization），并随后衍生出 霍夫曼编码（Huffman Coding）。

香农引入如下概念：

1. 失真率函数（distortion-rate function）：提供编码后信号失真的下界。
2. 矢量量化（vector quantization）

适定问题（well-posed problem），满足：

1. 解是唯一的；
2. 解是存在的；
3. 解依赖于输入数据在 reasonable topology。

使用有限多比特表示实数会产生 **舍入误差**。**截断误差** 因迭代算法只能执行有限次而产生。

### NN 量化

NN 量化的特殊之处：

1. NN 的推理和训练都是计算密集型（intensity）。
2. 因 NN 的重度参数化，有足够的机会降低 bit 表示而不影响精度。
3. NN 对 aggressive 量化和 extreme discretization 非常鲁棒。

## 量化基本概念

