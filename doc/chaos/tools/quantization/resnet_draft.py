import torch
from torchvision.models import quantization as models
from torch import nn

def create_combined_model(model_fe):
    # 步骤1：分离特征提取器
    model_fe_features = nn.Sequential(
        model_fe.quant,  # 量化 input
        model_fe.conv1,
        model_fe.bn1,
        model_fe.relu,
        model_fe.maxpool,
        model_fe.layer1,
        model_fe.layer2,
        model_fe.layer3,
        model_fe.layer4,
        model_fe.avgpool,
        model_fe.dequant,  # 反量化 output
    )

    # 步骤2：创建一个新的“头”
    new_head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 10),
    )

    # 步骤3：合并，不要忘记量化 stubs
    new_model = nn.Sequential(
        model_fe_features,
        nn.Flatten(1),
        new_head,
    )
    return new_model


# 注意 `quantize=False`
model = models.resnet18(pretrained=True, progress=True, quantize=False)
num_ftrs = model.fc.in_features

# Step 1
model.train()
model.fuse_model()
# Step 2
model_ft = create_combined_model(model)
# Use default QAT configuration
model_ft[0].qconfig = torch.quantization.default_qat_qconfig
# Step 3
model_ft = torch.quantization.prepare_qat(model_ft, inplace=True)

for param in model_ft.parameters():
  param.requires_grad = True

