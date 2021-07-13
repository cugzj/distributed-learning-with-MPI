from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from .utils import _get_dataset

def get_dataset(bsz, num_task):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(root='./dataset/mnist_data', train=True, download=True, transform=transform)
    testset = MNIST(root='./dataset/mnist_data', train=False, download=True, transform=transform)
    
    return _get_dataset(trainset, testset, list(range(num_task)), bsz)