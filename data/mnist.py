from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from utils import _get_dataset

def get_dataset(rank, bsz, workers, dataset_root='./dataset'):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(root=dataset_root + '/mnist_data', train=True, download=True, transform=transform)
    testset = MNIST(root=dataset_root + '/mnist_data', train=False, download=True, transform=transform)
    
    # _get_dataset(rank, dataset, workers, batch_size)
    return _get_dataset(rank, trainset, workers, bsz)

if __name__ == "__main__":
    get_dataset(1, 32, [1, 2, 3, 4, 5], 'D:\\dataset')