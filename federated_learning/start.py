# -*- coding: utf-8 -*-
from multiprocessing import process
import os, numpy
import torch
import torch.multiprocessing as mp

# Creat the parser 
import argparse
parser = argparse.ArgumentParser(description='Model Training')
# Method description
parser.add_argument('--method', type=str, default='FedAvg', help='Running algorithm')
method = parser.parse_known_args()[0].method

# Dataset 
parser.add_argument('--root', type=str, default='./dataset', help='The root of dataset')
parser.add_argument('--dataset', type=str, default='mnist', help='The name of dataset used')
parser.add_argument('--presplit', type=bool, default=False, help='Use the split dataset as training')
parser.add_argument('--non-iid', type=bool, default=False, help='The distribution of training data')
parser.add_argument('--dirichlet', type=bool, default=False, help='Non-iid distribution follows Dirichlet')
parser.add_argument('--dir-alpha', type=float, default=0.1, help='The alpha value for dirichlet distrition')
parser.add_argument('--pathological', type=bool, default=False, help='Non-iid distribution follows Pathological')
parser.add_argument('--classes', type=int, default=2, help='Number of classes on each client')

# Model 
parser.add_argument('--model', type=str, default='VGG19', help='The name of model used') 

# Result output root 
parser.add_argument('--result', type=str, default='./result', help='The directory of the result')

# Other settings
parser.add_argument('--bsz', type=int, default=64, help='Batch size for training dataset')
parser.add_argument('--partial', type=bool, default=False, help='Partial workers selection') 
parser.add_argument('--num-part', type=int, default=1, help='Number of partipants')
# parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs')
try:
    # Use MPI 
    from mpi4py import MPI
    mpi = True
    size = MPI.COMM_WORLD.Get_size()
    idex = MPI.COMM_WORLD.Get_rank()
    parser.add_argument('--num-workers',type=int, default=(size-1), help='Total number of workers')
    parser.add_argument('--multiprocessing', type=bool, default=False, help='Whether to use multiprocessing (not implemented yet)')
    parser.add_argument('--backend', type=str, default='mpi', help='MPI or GLOO')
except:
    # Use Multiprocessing 
    mpi = False
    parser.add_argument('--size', type=int, default=1, help='Number of Processes without server')
    parser.add_argument('--num-workers',type=int, default=1, help='Total number of workers')


import importlib
# import an algorithm's server and learner and arguments 
add_parser_arguments = importlib.import_module('{}.add_parser_arguments'.format(method))
param_server = importlib.import_module('{}.param_server'.format(method))
learner = importlib.import_module('{}.learner'.format(method))

# add arguments 
add_parser_arguments.new_arguements(parser)
args = parser.parse_args()

# import dataset and model 
import sys
sys.path.insert(1, '../')
dataset = importlib.import_module('data.{}'.format(args.dataset))
model = importlib.import_module('models.{}'.format(args.dataset))
model = getattr(model, args.model)()

workers = numpy.arange(args.num_workers) + 1  # workers' indices start with 1, server is always 0 

# Define CPU 
cpu = torch.device('cpu')

if mpi:
    gpu = torch.device('cuda:{}'.format(idex%torch.cuda.device_count())) if torch.cuda.is_available() else torch.device('cpu')
    print('Hello World MPI! I am process', idex, 'of', size)
    # Run with MPI
    if idex == 0:
        test_data = dataset.get_testdataset(args.root)
        param_server.init_processes(idex, size, model, args, test_data, cpu, gpu, args.backend.lower())
    else:
        ranks = numpy.array_split(workers, size-1)[idex-1]
        if args.presplit:
            data_ratio_pairs, _ = dataset.get_dataset_with_precat(ranks, workers, args.root)
        else:
            alpha = args.dir_alpha if args.dirichlet else args.classes
            data_ratio_pairs, _ = dataset.get_dataset(ranks, workers, args.non_iid, args.dirichlet, alpha, dataset_root=args.root)
        learner.init_processes(idex, ranks, size, model, args, data_ratio_pairs, cpu, gpu, args.backend.lower())
else:
    # Run with multiprocessing
    processes = []
    mp.set_start_method("spawn")
    for idex in range(args.size+1):
        gpu = torch.device('cuda:{}'.format(idex%torch.cuda.device_count())) if torch.cuda.is_available() else torch.device('cpu')
        print('Hello World Multiprocessing! I am process', idex, 'of', size)
        if idex == 0:
            test_data = dataset.get_testdataset(args.root)
            p = mp.Process(target=param_server.init_processes, args=(idex, size, model, args, test_data, cpu, gpu, args.backend.lower()))
        else:
            ranks = numpy.array_split(workers, size-1)[idex-1]
            if args.presplit:
                data_ratio_pairs, _ = dataset.get_dataset_with_precat(ranks, workers, args.root)
            else:
                alpha = args.dir_alpha if args.dirichlet else args.classes
                data_ratio_pairs, _ = dataset.get_dataset(ranks, workers, args.non_iid, args.dirichlet, alpha, dataset_root=args.root)
            p = mp.Process(target=learner.init_processes, args=(idex, ranks, size, model, args, data_ratio_pairs, cpu, gpu, args.backend.lower()))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    

# Usage: Different approaches should create dedicated param_server and learners 