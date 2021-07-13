from random import Random
from itertools import combinations, chain
import math
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset

class IID_DataPartitioner(object):

    def __init__(self, data, num_worker, sizes=None, seed=1234):
        rng = Random()
        rng.seed(seed)
        if sizes is None:
            sizes = [rng.random() for i in range(num_worker)]
            sizes = [s/sum(sizes) for s in sizes]
        self.data, self.partitions, self.weights = data, [], sizes
        data_len = len(data)
        indexes = [x for x in range(data_len)]
        rng.shuffle(indexes)

        for ratio in sizes:
            part_len = int(ratio * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

        lengths = [len(part) for part in self.partitions]
        self.weights = [l/sum(lengths) for l in lengths]

    def use(self, partition):
        return Subset(self.data, self.partitions[partition]), self.weights[partition]

def data_catogerize(data:Dataset, seed=1234):
    data_dict = {}
    for idx in range(len(data)):
        _, target = data.__getitem__(idx)
        if target not in data_dict:
            data_dict[target] = []
        data_dict[target].append(idx)
    rng = Random()
    rng.seed(seed)
    for key in data_dict.keys():
        rng.shuffle(data_dict[key])
    return data_dict

def worker_labels(targets, classes, num_workers, balanced=False, seed=1234):
    if classes != -1 and num_workers * classes < len(targets):
        raise Exception('Failed to generate a list of labels')
    labels = []
    rng = Random()
    rng.seed(seed)
    if len(targets) < num_workers and classes == -1:
        classes = 1
    if classes == -1:
        temp = [target for target in targets]
        for i in range(num_workers):
            a = int(len(temp)/(num_workers - i))
            labels.append(temp[0:a])
            temp = temp[a:]
        balanced = True if (len(targets) % num_workers == 0) else False
    else:
        if balanced and num_workers * classes % len(targets) == 0:
            temp = [target for target in targets] * (num_workers * classes // len(targets))
            for i in range(math.ceil(len(temp)/classes)):
                labels.append([temp[(i*classes+j) % len(temp)] for j in range(classes)])
            balanced = True
        else:
            temp = [target for target in targets]
            for i in range(math.ceil(len(temp)/classes)):
                labels.append([temp[(i*classes+j) % len(temp)] for j in range(classes)])
            a = list(combinations(temp, classes))
            labels += rng.choices(a, k=num_workers-len(labels))
            balanced = False
    
    return labels

class NonIID_DataPartitioner(object):

    def __init__(self, data, num_worker, sizes=None, classes=-1, seed=1234):
        self.data, self.partitions, self.weights = data, [], sizes
        data_dict = data_catogerize(data)
        rng = Random()
        rng.seed(seed)
        if sizes is not None:
            labels = worker_labels(data_dict.keys(), classes, num_worker, balanced=(min(sizes) == max(sizes)))
        else:
            labels = worker_labels(data_dict.keys(), classes, num_worker)
        shard_size = {target: len(samples)//(sum([(target in wl) for wl in labels])) for target, samples in data_dict.items()}
        
        for idx in range(num_worker):
            all_sets, partition = [], []
            for label in labels[idx]:
                a = shard_size[label]
                all_sets.append(data_dict[label][0:a])
                data_dict[label] = data_dict[label][a:]
            max_len = max([len(p) for p in all_sets])
            for i in range(max_len):
                for j in range(len(all_sets)):
                    if len(all_sets[j]) > i:
                        partition.append(all_sets[j][i])
            self.partitions.append(partition)

        if sizes is not None:
            lengths = [len(part) for part in self.partitions]
            min_length = min(min([int(l/sizes[idx]) for idx, l in enumerate(lengths)]), len(self.data))
            for idx, size in enumerate(sizes):
                self.partitions[idx] = self.partitions[idx][0:int(min_length*size)]
        lengths = [len(part) for part in self.partitions]
        self.weights = [l/sum(lengths) for l in lengths]
        
    def use(self, partition):
        return Subset(self.data, self.partitions[partition]), self.weights[partition]


def partition_dataset(dataset, workers, iid, balanced, random_weight, classes, p=(1.0, 0.0)):
    """ Partitioning Data """
    workers_num = len(workers)
    if balanced:
        partition_sizes = [1.0 * p[0] / workers_num for _ in range(workers_num)]
    elif random_weight:
        partition_sizes = None
    else: 
        partition_sizes = [float((i+1) * 2 / ((1+workers_num) * workers_num)) * p[0] for i in range(workers_num)]
    
    if iid:
        partition = IID_DataPartitioner(dataset, workers_num, partition_sizes)
    else:
        partition = NonIID_DataPartitioner(dataset, workers_num, partition_sizes, classes)
    return partition


def select_dataset(workers: list, rank: int, partition, batch_size: int):
    workers_num = len(workers)
    partition_dict = {workers[i]: i for i in range(workers_num)}
    partition, weight = partition.use(partition_dict[rank])
    return DataLoader(partition, batch_size=batch_size, shuffle=False), weight