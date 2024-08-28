import medmnist
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from medmnist import ChestMNIST, RetinaMNIST, SynapseMNIST3D, INFO, Evaluator

import warnings
from tqdm import tqdm

default_transform = Compose(
    [
        Resize((224, 224)),
        ToTensor(),
    ]
)

def setup ():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        warnings.warn("no CUDA devices found.", ResourceWarning)
        device = torch.device('cpu')

def load_datasets (data_flag, size=28, as_rgb=True, download=True, transform=default_transform, batch_size=64, shuffle=True,):
    dataset = INFO[data_flag]
    DataClass = getattr(medmnist, dataset['python_class'])
    attr_dict = {
            'task': dataset['task'],
            'num_classes': len(dataset['label'])
    }

    assert type(download) == bool, "Parameter `download` must of boolean type."
    assert type(as_rgb) == bool, "Parameter `as_rgb` must of boolean type."
    assert type(batch_size) == int, "Parameter `batch_size` must of integer type."
    assert type(shuffle) == bool, "Parameter `shuffle` must of boolean type."
    assert size in [28, 64, 128, 224], "Invalid size. Only 28, 64, 128 and 224 are accepted."
    
    train_dataset = DataClass(split="train", transform=transform, download=download, as_rgb=as_rgb, size=size,)
    val_dataset = DataClass(split="val", transform=transform, download=download, as_rgb=as_rgb, size=size,)
    test_dataset = DataClass(split="test", transform=transform, download=download, as_rgb=as_rgb, size=size,)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader, attr_dict 

def checkpoint ():
    pass