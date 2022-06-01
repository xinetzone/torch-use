# 高级教程

参考：[静态量化](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

本教程展示了如何进行训练后的静态量化，并说明了两个更高级的技术——逐通道量化和感知量化的训练——以进一步提高模型的准确性。注意，量化目前 CPU 支持更好，所以在本教程中我们不会使用 GPU/CUDA。在本教程结束时，您将看到 PyTorch 中的量化如何在提高速度的同时显著降低模型大小。此外，您还将看到如何轻松地应用这里所展示的一些高级量化技术，从而使您的量化模型比其他方法获得更少的精度。

```{toctree}
:maxdepth: 3

custom
run
qat
cifar
```