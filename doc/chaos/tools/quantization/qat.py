from typing import Any, Callable, ParamSpec, TypeVar
from importlib import import_module

import torch
import copy
from torch.ao.quantization import prepare_qat, get_default_qat_qconfig, convert

P = ParamSpec("P")
T = TypeVar('T')


class QuantizableCustom:
    '''
    Args:
        backend: 若是 x86，则为 ``'fbgemm'``，否则为 ``'qnnpack'``
        model_name: 模型名称，例如 'resnet18', 'resnet50', 'resnext101_32x8d',
            'mobilenet_v2', 'mobilenetv3', 'mobilenet_v3_large',
            'inception_v3', 'googlenet',
            'shufflenetv2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0'
    '''

    def __init__(self, model_name: str, backend: str = 'fbgemm'):
        self.model_name = model_name
        self.model_names = {
            'resnet18', 'resnet50', 'resnext101_32x8d',
            'mobilenet_v2', 'mobilenetv3', 'mobilenet_v3_large',
            'inception_v3', 'googlenet',
            'shufflenetv2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0'
        }
        assert self.model_name in self.model_names
        self.backend = backend

    def qconfig(self,
                quantize: bool = False,
                pretrained: bool = False,
                progress: bool = True,
                **kwargs: Any):
        """配置量化模型

        Args:
            pretrained: 若为 True，则返回在 ImageNet 预训练的模型
            progress: 若为 True，则显示下载到标准错误的进度条
            quantize: 若为 True，则返回模型的量化版本
    
        Return:
            配置后的模型
        """
        m = import_module('torchvision.models.quantization')
        mod_func = getattr(m, self.model_name)
        # 配置好的模型
        mod = mod_func(pretrained=pretrained,
                       progress=progress,
                       quantize=quantize,
                       **kwargs)
        mod.qconfig = get_default_qat_qconfig(self.backend)
        return mod

    def quantize_qat(self, model,
                     run_fn: Callable[P, T],
                     *,
                     run_args: ParamSpec,
                     run_kwargs: Any,
                     inplace: bool = False,
                     ):
        """QAT 并输出量化模型

        Args:
            run_fn: 用于评估准备好的模型的函数，可以是简单运行准备好的模型的函数，也可以是训练循环
            run_args: ``run_fn`` 的位置参数
            inplace: 就地进行模型转换，原始模块会发生改变

        Return:
            量化后的模型
        """
        torch._C._log_api_usage_once("quantization_api.quantize.quantize_qat")
        if not inplace:
            model = copy.deepcopy(model)
        model.train()
        prepare_qat(model, inplace=True)
        run_fn(model, *run_args, **run_kwargs)
        convert(model, inplace=True)
        return model
