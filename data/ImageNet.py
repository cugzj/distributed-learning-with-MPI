import os
from torchvision.datasets import ImageNet, ImageFolder
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms

from .utils import _get_partitioner, _use_partitioner

class myImageNet(ImageNet):
    # CAUTION: SET THE LINK BELOW TO EMPTY WHEN MADE PUBLIC 
    TRAIN_DOWNLOAD  = ""
    VAL_DOWNLOAD    = ""
    DEVKIT_DOWNLOAD = ""

    def __init__(self, root: str, train: bool=True, transform: transforms=None, target_transform: transforms=None, download: bool=False):
        if train and download:
            if os.path.exists(os.path.join(root, os.path.basename(self.TRAIN_DOWNLOAD))):
                print('Files already downloaded and verified')
            else:
                if self.TRAIN_DOWNLOAD == "" or self.TRAIN_DOWNLOAD is None:
                    raise Exception("The dataset is no longer publicly accessible. ")
                download_url(self.TRAIN_DOWNLOAD, self.root)
        
        if not train and download:
            if os.path.exists(os.path.join(root, os.path.basename(self.VAL_DOWNLOAD))):
                print('Files already downloaded and verified')
            else:
                if self.VAL_DOWNLOAD == "" or self.VAL_DOWNLOAD is None:
                    raise Exception("The dataset is no longer publicly accessible. ")
                download_url(self.VAL_DOWNLOAD, self.root)
        
        if download:
            if os.path.exists(os.path.join(root, os.path.basename(self.DEVKIT_DOWNLOAD))):
                print('Files already downloaded and verified')
            else:
                if self.DEVKIT_DOWNLOAD == "" or self.DEVKIT_DOWNLOAD is None:
                    raise Exception("The dataset is no longer publicly accessible. ")
                download_url(self.DEVKIT_DOWNLOAD, self.root)

        split = 'train' if train else 'val'
        super(myImageNet, self).__init__(root, split=split, download=download, transform=transform, target_transform=target_transform)



train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([ 
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataset(ranks:list, workers:list, isNonIID:bool, isDirichlet:bool=False, alpha=3, data_aug:bool=True, dataset_root='./dataset'):
    if data_aug:
        trainset = myImageNet(root=dataset_root + '/ImageNet', train=True, download=True, transform=train_transform)
        testset = myImageNet(root=dataset_root + '/ImageNet', train=False, download=True, transform=test_transform)
    else: 
        trainset = myImageNet(root=dataset_root + '/ImageNet', train=True, download=True)
        testset = myImageNet(root=dataset_root + '/ImageNet', train=False, download=True)
    
    partitioner = _get_partitioner(trainset, workers, isNonIID, isDirichlet, alpha)
    data_ratio_pairs = []
    for rank in ranks:
        data, ratio = _use_partitioner(partitioner, rank, workers)
        data_ratio_pairs.append((data, ratio))
    return data_ratio_pairs, testset

def get_dataset_with_precat(ranks:list, workers:list, dataset_root='./dataset'):
    testset = myImageNet(root=dataset_root + '/TinyImageNet', train=False, download=True, transform=test_transform)

    data_ratio_pairs = []
    for rank in ranks:
        idx = workers.index(rank)
        current_path = dataset_root + '/TinyImageNet/{}_partitions/{}'.format(len(workers), idx)
        trainset = ImageFolder(root=current_path, transform=train_transform)
        with open(current_path + '/weight.txt', 'r') as f:
            ratio = eval(f.read())
        data_ratio_pairs.append((trainset, ratio))
    
    return data_ratio_pairs, testset

if __name__ == "__main__":
    # store partitioned dataset 
    num_workers = 10
    workers = list(range(num_workers))
    path = 'D:/dataset'
    
    data_ratio_pairs, _ = get_dataset(workers, workers, isNonIID=False, dataset_root=path, data_aug=False)
    path = path + '/TinyImageNet/{}_partitions'.format(num_workers)
    if os.path.exists(path) is False:
        os.makedirs(path)

    for idx, pair in enumerate(data_ratio_pairs):
        data, ratio = pair
        current_path = os.path.join(path, str(idx))
        if os.path.exists(current_path) is False:
            os.makedirs(current_path)
        
        with open(current_path + '/weight.txt', 'w') as f:
            f.write(ratio)
        
        for i in range(len(data)):
            sample, target = data[i]
            if os.path.exists(os.path.join(current_path, str(int(target)))) is False:
                os.makedirs(os.path.join(current_path, str(int(target))))
            sample.save(current_path + '/{}/{}.jpg'.format(target, i))