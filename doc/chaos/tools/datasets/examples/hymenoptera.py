from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from ..imagenet import Transforms
from ..file import ZipDataset, PathType


class Hymenoptera:
    '''`Hymenoptera 数据集 <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`__ 是 ImageNet 的一个非常小的子集。

    有大约 120 个蚂蚁和蜜蜂的训练图像。每个类有 75 张验证图片。

    Args:
        data_dir: 数据根目录

    Attributes:
        dataset_sizes: 数据集大小
        dataloaders: 数据加载器
        class_names: 类名列表
    '''

    def __init__(self, data_dir: PathType='data/hymenoptera_data',
                 download: bool = False) -> None:
        data_dir = Path(data_dir)
        if download or not data_dir.exists():
            self.loader()
        data_types = ['train', 'val']
        image_datasets = {x: ImageFolder(data_dir/x, getattr(Transforms, x))
                          for x in data_types}
        self.dataloaders = {x: DataLoader(image_datasets[x],
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4)
                            for x in data_types}
        self.dataset_sizes = {x: len(image_datasets[x]) for x in data_types}
        self.class_names = image_datasets['train'].classes

    def loader(self):
        root = 'data'
        url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
        zip_name = url.split('/')[-1]
        zipset = ZipDataset(root)
        zip_name = zipset.download(url, zip_name)  # 下载数据
        zipset.extractall(zip_name)  # 解压数据
