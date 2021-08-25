import os
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as transforms

from .utils import _get_partitioner, _use_partitioner

class TinyImagenet(ImageFolder):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    # CAUTION: SET THE LINK BELOW TO EMPTY WHEN MADE PUBLIC 
    DOWNLOAD = ""

    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False):
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.download = download

        if download:
            if os.path.exists(os.path.join(root, os.path.basename(self.DOWNLOAD))):
                print('Files already downloaded and verified')
            else:
                if self.DOWNLOAD == "" or self.DOWNLOAD is None:
                    raise Exception("The dataset is no longer publicly accessible. ")
                download_and_extract_archive(self.DOWNLOAD, self.root)
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

def get_dataset(ranks:list, workers:list, isNonIID:bool, isDirichlet:bool=False, alpha=3, data_aug:bool=True, dataset_root='./dataset'):
    if data_aug:
        trainset = TinyImagenet(root=dataset_root + '/TinyImageNet', train=True, download=True, transform=train_transform)
        testset = TinyImagenet(root=dataset_root + '/TinyImageNet', train=False, download=True, transform=test_transform)
    else: 
        trainset = TinyImagenet(root=dataset_root + '/TinyImageNet', train=True, download=True)
        testset = TinyImagenet(root=dataset_root + '/TinyImageNet', train=False, download=True)
    
    partitioner = _get_partitioner(trainset, workers, isNonIID, isDirichlet, alpha)
    data_ratio_pairs = {}
    for rank in ranks:
        data, ratio = _use_partitioner(partitioner, rank, workers)
        data_ratio_pairs[rank] = (data, ratio)
    return data_ratio_pairs, testset

def get_dataset_with_precat(ranks:list, workers:list, dataset_root='./dataset'):
    testset = TinyImagenet(root=dataset_root + '/TinyImageNet', train=False, download=True, transform=test_transform)

    data_ratio_pairs = {}
    for rank in ranks:
        idx = np.where(workers == rank)[0][0]
        current_path = dataset_root + '/TinyImageNet/{}_partitions/{}'.format(len(workers), idx)
        trainset = ImageFolder(root=current_path, transform=train_transform)
        with open(current_path + '/weight.txt', 'r') as f:
            ratio = eval(f.read())
        data_ratio_pairs[rank] = (trainset, ratio)
    
    return data_ratio_pairs, testset

def get_testdataset(dataset_root='./dataset'):
    testset = TinyImagenet(root=dataset_root + '/TinyImageNet', train=False, download=True, transform=test_transform)
    return testset

if __name__ == "__main__":
    # store partitioned dataset 
    num_workers = 10
    workers = np.arange(num_workers) + 1
    path = 'D:/dataset'
    
    data_ratio_pairs, _ = get_dataset(workers, workers, isNonIID=False, dataset_root=path, data_aug=False)
    path = path + '/TinyImageNet/{}_partitions'.format(num_workers)
    if os.path.exists(path) is False:
        os.makedirs(path)

    for idx, pair in data_ratio_pairs.items():
        data, ratio = pair
        current_path = os.path.join(path, str(idx))
        if os.path.exists(current_path):
            import shutil
            shutil.rmtree(current_path)
        os.makedirs(current_path)

        with open(current_path + '/weight.txt', 'w') as f:
            f.write('{}\t{}\n'.format(idx, ratio))
        
        for i in range(len(data)):
            sample, target = data[i]
            if os.path.exists(os.path.join(current_path, str(int(target)))) is False:
                os.makedirs(os.path.join(current_path, str(int(target))))
            sample.save(current_path + '/{}/{}.jpg'.format(target, i))