import numpy as np
import torch
from torch._C import device
from torch.optim import SGD
import torch.distributed as dist
from torch.autograd import Variable
from copy import deepcopy

def update_model(model, aggregated_model, cpu, gpu):
    # all_param = model.state_dict()
    
    # send parameters to PS
    for param in aggregated_model.parameters():
        param_cpu = torch.tensor(param.data, device=cpu)
        dist.gather(tensor=param_cpu, dst=0)
    
    # receive parameters from PS
    for param in model.parameters():
        recv = torch.zeros_like(param.data, device=cpu)
        dist.scatter(tensor=recv, src=0)
        param.data = torch.tensor(recv.data, device=gpu)

    # model.load_state_dict(all_param)

def run(workers, size, model, args, data_ratio_pairs:dict, cpu, gpu):
    # Send the weights to server 
    weights = [w for _, w in data_ratio_pairs.values()]
    dist.gather(tensor=torch.tensor(weights), dst=0)

    model = model.cuda(gpu)
    # iterator = iter(train_data)
    iterators = [iter(train_data) for train_data, _ in data_ratio_pairs.values()]
    
    # workers = [v+1 for v in range(size-1)]
    # _group = [w for w in workers].append(rank)
    # group = dist.new_group(_group)

    # Receive initial model from server
    for idx, p in enumerate(model.parameters()):
        tmp_p = torch.zeros_like(p, device=cpu)
        dist.scatter(tensor=tmp_p, src=0)
        p.data = torch.tensor(tmp_p, device=gpu)

    print('Worker {} successfully received the model. '.format(list(workers)))

    for t in range(args.T):
        # Receive participants list 
        part_list = torch.zeros(size)
        dist.scatter(tensor=part_list, src=0)
        part_list = part_list.numpy()

        aggregated_loss  = 0.0
        aggregated_model = [torch.zeros_like(param, device=gpu) for param in model.parameters()]

        for idx, worker in enumerate(workers):
            if worker in part_list:
                mymodel = deepcopy(model)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = SGD(mymodel.parameters(), lr=args.lr)
                tot_loss = 0.0

                # perform local update 
                for _ in range(args.K):
                    try:
                        data, target = next(iterators[idx])
                    except:
                        iterators[idx] = iter(data_ratio_pairs[worker][0])
                        data, target = next(iterators[idx])
                    data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
                    optimizer.zero_grad()
                    output = mymodel(data)
                    loss = criterion(output, target)
                    tot_loss = tot_loss + loss.data / args.K
                    loss.backward()
                    optimizer.step()
                
                print('Worker: {}   Communition Rounds: {}    Loss: {}'.format(worker, t, tot_loss))
                weight = np.count_nonzero(part_list==worker) / len(part_list) if args.partial else weights[idx]
                aggregated_loss = aggregated_loss + tot_loss * weight
                aggregated_model = [aggregated_model[idx] + (param.data * weight) for idx, param in enumerate(mymodel.parameters())]

        loss_cpu = torch.tensor(aggregated_loss, device=cpu)
        dist.gather(tensor=loss_cpu, dst=0)

        update_model(model, aggregated_model, cpu, gpu)
        

def init_processes(rank, workers, size, model, args, data_ratio_pairs, cpu, gpu, backend='mpi'):
    dist.init_process_group(backend, rank=rank, world_size=size)
    run(workers, size, model, args, data_ratio_pairs, cpu, gpu)