# 概述

在较低的层次上，PyTorch 提供了一种表示量化张量并对它们执行操作的方法。它们可以用来直接构建模型，以较低的精度执行全部或部分计算。提供了更高层次的 API，结合了 FP32 模型转换的典型工作流程，以最小的精度损失降低精度。

量化需要用户了解三个概念：

- QConfig：指定量化 Weight 和 Activation 的配置方式
- 后端：用于支持量化的内核
- 引擎（{mod}`~torch.backends.quantized.engine`）：指定执行时所需后端

```{note}
在准备量化模型时，必须确保 `qconfig` 和用于量化计算的引擎与将在其上执行模型的后端匹配。
```

量化的后端：

- AVX2 X86 CPU：[`'fbgemm'`](https://github.com/pytorch/FBGEMM)
- ARM CPU（常用于手机和嵌入式设备）：[`'qnnpack'`](https://github.com/pytorch/QNNPACK)

相应的实现会根据 PyTorch 构建模式自动选择，不过用户可以通过将 `torch.backends.quantization.engine` 设置为 `'fbgemm'` 或 `'qnnpack'` 来覆盖这个选项。

量化感知训练（通过 {mod}`~torch.quantization.FakeQuantize`，它模拟 FP32 中的量化数字）支持 CPU 和 CUDA。

`qconfig` 控制量化传递期间使用的观测器器类型。当对线性和卷积函数和模块进行打包权重时，`qengine` 控制是使用 `fbgemm` 还是 `qnnpack` 特定的打包函数。例如：

```python
# set the qconfig for PTQ
qconfig = torch.quantization.get_default_qconfig('fbgemm')
# or, set the qconfig for QAT
qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
# set the qengine to control weight packing
torch.backends.quantized.engine = 'fbgemm'
```

## API 概述

PyTorch 提供了两种不同的量化模式：Eager 模式量化和 FX 图模式量化。

Eager 模式量化是一个 beta 特性。用户需要进行融合，并手动指定量化和反量化发生的位置，而且它只支持模块而不支持函数。

FX 图模式量化是 PyTorch 中一个新的自动量化框架，目前它是一个原型特性。它通过添加对函数的支持和量化过程的自动化，对 Eager 模式量化进行了改进，尽管人们可能需要重构模型，以使模型与 FX Graph 模式量化兼容（通过 `torch.fx` 符号可追溯）。注意 FX 图模式量化预计不会在任意工作模型由于模型可能不是符号可追溯，我们会将其集成到域库 torchvision 和用户将能够量化模型类似于支持域的库与 FX 图模式量化。对于任意的模型，我们将提供一般的指导方针，但要让它实际工作，用户可能需要熟悉 `torch.fx`，特别是如何使模型具有符号可追溯性。

新用户的量化鼓励首先尝试 FX 图模式量化，如果它不工作，用户可以尝试遵循[使用 FX 图模式量化的指导方针](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)或回落到 Eager 模式量化。

支持三种类型的量化：

1. 动态量化（通过读取/存储在浮点数中的激活量化权重，并量化用于计算。）
2. 静态量化（权重量化，激活量化，校准所需的后训练）
3. 静态量化感知训练（权重量化、激活量化、训练时建模的量化数值）
