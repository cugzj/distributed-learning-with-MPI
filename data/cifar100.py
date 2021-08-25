from torchvision.datasets import CIFAR100
import numpy as np
from torchvision.datasets.folder import ImageFolder
import torchvision.transforms as transforms
import os

from .utils import _get_partitioner, _use_partitioner

train_transform = transforms.Compose([ 
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
])
test_transform = transforms.Compose([ 
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
])

def get_dataset(ranks:list, workers:list, isNonIID:bool, isDirichlet:bool=False, alpha=3, data_aug:bool=True, dataset_root='./dataset'):
    if data_aug:
        trainset = CIFAR100(dataset_root + '/cifar100_data', train=True, download=True, transform=train_transform)
        testset = CIFAR100(dataset_root + '/cifar100_data', train=False, download=True, transform=test_transform)
    else: 
        trainset = CIFAR100(dataset_root + '/cifar100_data', train=True, download=True)
        testset = CIFAR100(dataset_root + '/cifar100_data', train=False, download=True)

    partitioner = _get_partitioner(trainset, workers, isNonIID, isDirichlet, alpha)
    data_ratio_pairs = {}
    for rank in ranks:
        data, ratio = _use_partitioner(partitioner, rank, workers)
        data_ratio_pairs[rank] = (data, ratio)
    return data_ratio_pairs, testset

def get_dataset_with_precat(ranks:list, workers:list, dataset_root='./dataset'):
    testset = CIFAR100(root=dataset_root + '/cifar100_data', train=False, download=True, transform=test_transform)

    data_ratio_pairs = {}
    for rank in ranks:
        idx = np.where(workers == rank)[0][0]
        current_path = dataset_root + '/cifar100_data/{}_partitions/{}'.format(len(workers), idx)
        trainset = ImageFolder(root=current_path, transform=train_transform)
        with open(current_path + '/weight.txt', 'r') as f:
            ratio = eval(f.read())
        data_ratio_pairs[rank] = (trainset, ratio)
    
    return data_ratio_pairs, testset

def get_testdataset(dataset_root='./dataset'):
    testset = CIFAR100(root=dataset_root + '/cifar100_data', train=False, download=True, transform=test_transform)
    return testset

if __name__ == "__main__":
    # store partitioned dataset 
    num_workers = 10
    workers = np.arange(num_workers) + 1
    path = 'D:/dataset'
    
    data_ratio_pairs, _ = get_dataset(workers, workers, isNonIID=False, dataset_root=path, data_aug=False)
    path = path + '/cifar100_data/{}_partitions'.format(num_workers)
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