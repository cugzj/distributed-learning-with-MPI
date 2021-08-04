import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class PMNIST(MNIST):
    def __init__(self, source='./dataset/mnist_data', train = True, shuffle_seed = None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        super(PMNIST, self).__init__(source, train, download=True, transform=transform)
        
        self.train = train
        self.num_data = 0
        
        self.permuted_data = torch.stack(
            [img.type(dtype=torch.float32).view(-1)[shuffle_seed].view(1, 28, 28) / 255.0
                for img in self.data])
        self.num_data = self.permuted_data.shape[0]
            
    def __getitem__(self, index):
        input, label = self.permuted_data[index], self.targets[index] 
        return input, label

    def getNumData(self):
        return self.num_data

def get_dataset(bsz, num_task):
    train_loader, test_loader, labels = {}, {}, {}
    
    train_data_num, test_data_num = 0, 0
    
    np.random.seed(1234)
    for i in range(num_task):
        shuffle_seed = np.arange(28*28)
        if i > 0:
            np.random.shuffle(shuffle_seed)
        
        train_PMNIST_DataLoader = PMNIST(train=True, shuffle_seed=shuffle_seed)
        test_PMNIST_DataLoader = PMNIST(train=False, shuffle_seed=shuffle_seed)
        
        train_data_num += train_PMNIST_DataLoader.getNumData()
        test_data_num += test_PMNIST_DataLoader.getNumData()
        
        train_loader[i] = DataLoader(train_PMNIST_DataLoader, batch_size=bsz, shuffle=False)
        test_loader[i] = DataLoader(test_PMNIST_DataLoader, batch_size=bsz, shuffle=False)
        # labels[i] = train_PMNIST_DataLoader.targets
        labels[i] = list(range(10))
    
    # return train_loader, test_loader, int(train_data_num/num_task), int(test_data_num/num_task)
    return train_loader, test_loader, labels
