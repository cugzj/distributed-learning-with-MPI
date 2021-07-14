from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

from utils import _get_dataset

def get_dataset(bsz, num_task, dataset_root='./dataset'):
    # mean=[x/255 for x in [125.3,123.0,113.9]]
    # std=[x/255 for x in [63.0,62.1,66.7]]

    # transform = transforms.Compose([
    #     transforms.ToTensor(), 
    #     transforms.Normalize(mean, std)
    # ])

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
    

    trainset = CIFAR100(dataset_root + '/cifar100_data', train=True, download=True, transform=train_transform)
    testset = CIFAR100(dataset_root + '/cifar100_data', train=False, download=True, transform=test_transform)

    return _get_dataset(trainset, testset, list(range(num_task)), bsz)

if __name__ == "__main__":
    get_dataset(32, 5, 'D:\\dataset')