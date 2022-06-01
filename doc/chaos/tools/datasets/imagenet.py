from torch.utils import data as tdata
from torchvision.datasets import ImageNet
from torchvision import transforms


class Transforms:
    _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize
    ])
    val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        _normalize
    ])


def prepare_data_loaders(data_path, train_batch_size, eval_batch_size):
    dataset = ImageNet(data_path, split="train", transform=Transforms.train)
    dataset_test = ImageNet(data_path, split="val", transform=Transforms.val)

    train_sampler = tdata.RandomSampler(dataset)
    test_sampler = tdata.SequentialSampler(dataset_test)

    data_loader = tdata.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = tdata.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)
    return data_loader, data_loader_test
