import numpy as np
import os
from PIL import Image

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as transforms

from .utils import _get_dataset

class TinyImagenet(ImageFolder):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False):
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Files already downloaded and verified')
            else:
                url = 'http://www.image-net.org/data/tiny-imagenet-200.zip'
                filename = url.rpartition('/')[2]
                download_and_extract_archive(url, self.root)
                print('Processing...')

        self.data, self.targets = [], []
        wnids = os.listdir(os.path.join(self.root, 'tiny-imagenet-200/train'))
        if not train and os.path.exists(os.path.join(self.root, 'tiny-imagenet-200/val/images')):
            # format "/val" files 
            for idx in wnids:
                os.mkdir(path=os.path.join(self.root, 'tiny-imagenet-200/val/{}'.format(idx)))
            f = open(os.path.join(self.root, 'tiny-imagenet-200/val/val_annotations.txt'), 'r')
            for line in f.readlines():
                seq = line.split('\t')
                fname, idx = seq[0], seq[1]
                if idx in wnids:
                    src = os.path.join(self.root, 'tiny-imagenet-200/val/images/{}'.format(fname))
                    dst = os.path.join(self.root, 'tiny-imagenet-200/val/{}/{}'.format(idx, fname))
                    os.rename(src,dst)
            f.close()
            os.rmdir(os.path.join(self.root, 'tiny-imagenet-200/val/images'))
            os.remove(os.path.join(self.root, 'tiny-imagenet-200/val/val_annotations.txt'))
        
        root = os.path.join(root, 'tiny-imagenet-200/{}'.format('train' if train else 'val'))
        super(TinyImagenet, self).__init__(root, transform=transform, target_transform=target_transform)


def get_dataset(bsz, num_task):
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))
    ])
    test_transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)), 
    ])
    
    trainset = TinyImagenet('./dataset/TinyImageNet', train=True, download=True, transform=train_transform)
    testset = TinyImagenet('./dataset/TinyImageNet', train=False, download=True, transform=test_transform)
    
    return _get_dataset(trainset, testset, list(range(num_task)), bsz)

