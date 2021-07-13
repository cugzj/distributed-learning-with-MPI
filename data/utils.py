from random import Random
from torch.utils.data import DataLoader, Subset, Dataset

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

def worker_labels(targets, num_workers, seed=1234):
    labels, temp = [], [target for target in targets]

    for i in range(num_workers):
        a = int(len(temp)/(num_workers - i))
        labels.append(temp[0:a])  
        temp = temp[a:]  
    return labels 

class NonIID_DataPartitioner(object):

    def __init__(self, train_data, test_data, sizes, classes=-1, seed=1234):
        self.train_data, self.test_data = train_data, test_data
        self.train_partitions, self.test_partitions = [], []
        train_data_dict, test_data_dict = data_catogerize(train_data), data_catogerize(test_data)
        rng = Random()
        rng.seed(seed)
        self.labels = labels = worker_labels(sorted(train_data_dict.keys()), len(sizes))
        # self.labels = labels = worker_labels(train_data_dict.keys(), len(sizes))

        for idx, ratio in enumerate(sizes): 
            part_train_len, part_test_len = int(ratio * len(train_data)), int(ratio * len(test_data))
            train_partition, test_partition = [], []
            for j, label in enumerate(labels[idx]):
                a = int(part_train_len / (len(labels[idx]) - j)) 
                b = int(part_test_len / (len(labels[idx]) - j))
                train_partition.extend(train_data_dict[label][0:a])
                test_partition.extend(test_data_dict[label][0:b])
                train_data_dict[label], test_data_dict[label] = train_data_dict[label][a:], test_data_dict[label][b:]
                part_train_len, part_test_len = part_train_len - a, part_test_len - b
            rng.shuffle(train_partition)
            rng.shuffle(test_partition)
            self.train_partitions.append(train_partition)
            self.test_partitions.append(test_partition)
        
    def use(self, partition):
        return Subset(self.train_data, self.train_partitions[partition]), Subset(self.test_data, self.test_partitions[partition]), self.labels[partition]

def _get_dataset(train_dataset, test_dataset, workers, batch_size):
    """ Partitioning Data """
    workers_num = len(workers)
    partition_sizes = [1.0 / workers_num for _ in range(workers_num)]

    partition = NonIID_DataPartitioner(train_dataset, test_dataset, partition_sizes)
    
    train_loader, test_loader, labels = {}, {}, {}
    for i in workers:
        train_data, test_data, labels[i] = partition.use(i)
        train_loader[i] = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_loader[i]  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, labels