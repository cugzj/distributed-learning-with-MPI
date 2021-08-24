import time, os, json
import numpy as np

import torch
from torch._C import device
import torch.distributed as dist
from torch.autograd import Variable

def test_model(model, test_data, dev):
    correct, total = 0, 0
    model.eval()

    with torch.no_grad():
        for data, target in test_data:
            data, target = Variable(data).cuda(dev), Variable(target).cuda(dev)
            output = model(data)
            # get the index of the max log-probability
            _, predictions = output.max(1)
            total += predictions.size(0)
            correct += torch.sum(predictions == target.data).float()

    acc = correct / total
    return acc

def update_model(model, size, cpu, gpu):
    # all_param = model.state_dict()

    # receive the parameters from workers 
    for param in model.parameters():
        tensor = torch.zeros_like(param.data, device=cpu)
        gather_list = [torch.zeros_like(param.data, device=cpu) for _ in range(size)]
        dist.gather(tensor=tensor, gather_list=gather_list, dst=0)
        param.data = torch.zeros_like(param.data, device=gpu)
        for w in range(size):
            # Suppose the model received from clients are well processed 
            param.data = param.data + torch.tensor(gather_list[w].data, device=gpu)

    # send the parameters to workers 
    for param in model.parameters():
        tmp_p = torch.tensor(param.data, device=cpu)
        scatter_p_list = [tmp_p for _ in range(size)]
        dist.scatter(tensor=tmp_p, scatter_list=scatter_p_list)

    # model.load_state_dict(all_param)

# def update_model_full(model, size, cpu, gpu, weights):
#     all_param = model.state_dict()
#     weights_iter = iter(weights)

#     # receive the parameters from workers 
#     for key, param in all_param.items():
#         tensor = torch.zeros_like(param.data, device=cpu)
#         gather_list = [torch.zeros_like(param.data, device=cpu) for _ in range(size)]
#         dist.gather(tensor=tensor, gather_list=gather_list)
#         all_param[key].data = torch.zeros_like(param.data, device=gpu)

#         for w in range(size):
#             for local_param in gather_list[w]:
#                 weight = next(weights_iter)
#                 all_param[key].data = all_param[key].data + weight * torch.tensor(local_param.data, device=gpu)

#     # send the parameters to workers 
#     for param in all_param.values():
#         tmp_p = torch.tensor(param.data, device=cpu)
#         scatter_p_list = [tmp_p for _ in range(size)]
#         dist.scatter(tensor=tmp_p, scatter_list=scatter_p_list)

#     model.load_state_dict(all_param)

def run(size, model, args, test_data, f_result, cpu, gpu):
    # Receive the weights from all clients 
    weights = [torch.tensor([0.0]) for _ in range(size)]
    dist.gather(tensor=torch.tensor([0.0]), gather_list=weights, dst=0)
    weights = np.concatenate([list(w) for w in weights])
    print('weights:', weights)
    
    start = time.time()
    model = model.cuda(gpu)

    # # workers = [v+1 for v in range(size-1)]
    # # _group = [w for w in workers].append(rank)
    # # group = dist.new_group(_group)

    for p in model.parameters():
        tmp_p = torch.tensor(p.data, device=cpu)
        scatter_p_list = [tmp_p for _ in range(size)]
        # dist.scatter(tensor=tmp_p, scatter_list=scatter_p_list, group=group)
        dist.scatter(tensor=tmp_p, scatter_list=scatter_p_list)

    print('Model has sent to all nodes! ')
    print('Begin!') 

    for t in range(args.T):
        # send participants to all clients 
        participants = np.random.choice(np.arange(size), size=args.num_part, replace=True, p=weights) if args.partial else np.arange(size)
        print('Participants list:', list(participants))
        part_list = [torch.tensor(participants, device=cpu) for _ in range(size)]
        tmp_p = torch.tensor(participants, device=cpu)
        dist.scatter(tensor=tmp_p, scatter_list=part_list)

        # receive the list of train loss from workers
        info_list = [torch.tensor([0.0]) for _ in range(size)]
        # dist.gather(tensor=torch.tensor([0.0]), gather_list=info_list, group=group)
        dist.gather(tensor=torch.tensor([0.0]), gather_list=info_list, dst=0)
        # info_list = np.concatenate([list(a) for a in info_list])
        # train_loss = sum(info_list).item() / args.num_part if args.partial else sum(info_list * weights).item()
        train_loss = sum(info_list).item()

        # if args.partial:
        #     update_model_partial(model, size, cpu, gpu, args.num_part)
        # else:
        #     update_model_full(model, size, cpu, gpu, weights)
        update_model(model, size, cpu, gpu)

        timestamp = time.time() - start
        test_acc = test_model(model, test_data, gpu)
        print("Epoch: {}\t\tLoss: {}\t\tAccuracy: {}".format(t, train_loss, test_acc))
        f_result.write(str(t) + "\t" + str(timestamp) + "\t" + str(train_loss) + "\t" + str(test_acc) + "\n")
        f_result.flush()

def init_processes(rank, size, model, args, test_data, cpu, gpu, backend='mpi'):
    dist.init_process_group(backend, rank=rank, world_size=size)
    if not os.path.exists(args.result):
        os.makedirs(args.result)
    result_file = os.path.join(args.result, '{}.txt'.format(len(os.listdir(args.result))))
    f_result = open(result_file, 'w')
    f_result.write(json.dumps(args) + '\n')
    run(size, model, args, test_data, f_result, cpu, gpu)