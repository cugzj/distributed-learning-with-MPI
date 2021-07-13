# -*- coding: utf-8 -*-
import os, sys, time, math
import numpy as np

size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

# Required PyTorch Module 
import torch
from torch.autograd import Variable
from torch.multiprocessing import Process as TorchProcess
from torch.utils.data import DataLoader
import models.cifar, models.mnist
from data import select_dataset, partition_dataset

# Parameter Server and Learner 
import param_server, learner


import argparse
parser = argparse.ArgumentParser()
# Training info
parser.add_argument('--method', type=str, default='FedAvg')

# model and datasets 
parser.add_argument('--data-dir', type=str, default='./dataset')
parser.add_argument('--model', type=str, default='ResNet18OnCifar10')
parser.add_argument('--path', type=str, default='./')
parser.add_argument('--num-gpu', type=int, default=2)
parser.add_argument('--iid', type=bool, default=False)
parser.add_argument('--classes', type=int, default=-1) # the classes of each worker 
parser.add_argument('--balanced', type=bool, default=False)  # whether the data volume among the workers are the same
parser.add_argument('--random-weight', type=bool, default=False)

# Hyper parameters setting 
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--train-bsz', type=int, default=25) # Individual batch size
parser.add_argument('--E', type=int, default=128) # number of local iterations 
parser.add_argument('--T', type=int, default=2000) # number of global synchronizations 

args = parser.parse_args()

# print('Begin get models!')
""" Get model and train/test datasets """
if args.model == 'MnistCNN':
    model = models.mnist.CNN() 
    train_dataset, test_dataset = models.mnist.load_mnist_datasets(args.data_dir)

elif args.model == 'LeNetOnMnist':
    model = models.mnist.LeNet()
    train_dataset, test_dataset = models.mnist.load_mnist_datasets(args.data_dir)

elif args.model == 'LROnMnist':
    model = models.mnist.LR()
    train_dataset, test_dataset = models.mnist.load_mnist_datasets(args.data_dir)

elif args.model == 'LROnCifar10':
    model = models.cifar.LR(10)
    train_dataset, test_dataset = models.cifar.load_cifar_datasets(args.data_dir, n_class=10)

elif args.model == 'AlexNetOnCifar10':
    model = models.cifar.AlexNet(10)
    train_dataset, test_dataset = models.cifar.load_cifar_datasets(args.data_dir, n_class=10)

elif args.model == 'AlexNetOnCifar100':
    model = models.cifar.AlexNet(100)
    train_dataset, test_dataset = models.cifar.load_cifar_datasets(args.data_dir, n_class=100)
        
elif args.model == 'ResNet18OnCifar10':
    model = models.cifar.ResNet18(10)
    train_dataset, test_dataset = models.cifar.load_cifar_datasets(args.data_dir, n_class=10)

elif args.model == 'VGG11OnCifar10':
    model = models.cifar.vgg11(10)
    train_dataset, test_dataset = models.cifar.load_cifar_datasets(args.data_dir, n_class=10)

elif args.model == 'VGG16OnCifar10':
    model = models.cifar.vgg16(10)
    train_dataset, test_dataset = models.cifar.load_cifar_datasets(args.data_dir, n_class=10)

elif args.model == 'VGG11OnCifar100':
    model = models.cifar.vgg11(100)
    train_dataset, test_dataset = models.cifar.load_cifar_datasets(args.data_dir, n_class=100)

else:
    print('Model is not found!')
    sys.exit(-1)

"""  Start and Run  """
if rank == 0:
    # This is parameter server 
    test_data = DataLoader(test_dataset, batch_size=min(1000, len(test_dataset)), shuffle=False)
    param_server.init_processes(rank, size, model, args, test_data)
else:
    # This is worker nodes
    workers = [v+1 for v in range(size-1)]
    partitioner = partition_dataset(train_dataset, workers, args.iid, args.balanced, args.random_weight, args.classes)
    train_data, weight = select_dataset(workers, rank, partitioner, batch_size=args.train_bsz)
    print('Rank {} weight: {} '.format(rank, weight))
    learner.init_processes(rank, size, model, args, train_data, weight)

print('Rank {} finishes. '.format(rank))

