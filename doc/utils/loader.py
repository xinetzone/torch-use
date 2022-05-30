# import numpy as np
# from torch.utils.data import DataLoader
# from torchvision import datasets
from torch import device, cuda

device = device("cuda:0" if cuda.is_available() else "cpu")