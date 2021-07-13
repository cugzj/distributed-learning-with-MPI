import time, os

import torch
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

    acc = format(correct / total, '.4%')
    return acc

def update_model(model, size, group, cpu, gpu):
    all_param = model.state_dict()

    # receive the parameters from workers 
    for key, param in all_param.items():
        tensor = torch.zeros_like(param.data, device=cpu)
        gather_list = [torch.zeros_like(param.data, device=cpu) for _ in range(size)]
        dist.gather(tensor=tensor, gather_list=gather_list, group=group)
        all_param[key].data = torch.zeros_like(param.data, device=gpu)
        for w in range(size):
            all_param[key].data = all_param[key].data + torch.tensor(gather_list[w].data, device=gpu)

    # send the parameters to workers 
    for param in all_param.values():
        tmp_p = torch.tensor(param.data, device=cpu)
        scatter_p_list = [tmp_p for _ in range(size)]
        dist.scatter(tensor=tmp_p, scatter_list=scatter_p_list, group=group)

    model.load_state_dict(all_param)

def run(rank, size, model, args, test_data, f_result):
    cpu, gpu = torch.device('cpu'), torch.device('cuda:{}'.format(rank%args.num_gpu))

    start = time.time()
    model = model.cuda(gpu)

    workers = [v+1 for v in range(size-1)]
    _group = [w for w in workers].append(rank)
    group = dist.new_group(_group)

    for p in model.parameters():
        tmp_p = torch.tensor(p.data, device=cpu)
        scatter_p_list = [tmp_p for _ in range(size)]
        dist.scatter(tensor=tmp_p, scatter_list=scatter_p_list, group=group)

    print('Model has sent to all nodes! ')
    print('Begin!') 

    for t in range(args.T):
        # receive the list of train loss from workers
        info_list = [torch.tensor([0.0]) for _ in range(len(workers) + 1)]
        dist.gather(tensor=torch.tensor([0.0]), gather_list=info_list, group=group)
        train_loss = sum(info_list).item()

        update_model(model, size, group, cpu, gpu)

        timestamp = time.time() - start
        test_acc = test_model(model, test_data, gpu)
        print("Epoch: {}\t\tLoss: {}\t\tAccuracy: {}".format(t, train_loss, test_acc))
        f_result.write(str(t) + "\t" + str(timestamp) + "\t" + str(train_loss) + "\t" + str(test_acc) + "\n")
        f_result.flush()

def init_processes(rank, size, model, args, test_data, backend='mpi'):
    dist.init_process_group(backend, rank=rank, world_size=size)
    if os.path.isdir(args.path + 'result/') is False:
        os.mkdir(args.path + 'result/')
    if args.iid:
        if args.balanced:
            result_file = args.path + 'result/' + args.method + '_' + args.model + '_iid_balanced.out'
        else:
            result_file = args.path + 'result/' + args.method + '_' + args.model + '_iid_unbalanced.out'
    else:
        if args.balanced:
            result_file = args.path + 'result/' + args.method + '_' + args.model + '_noniid_balanced.out'
        else:
            result_file = args.path + 'result/' + args.method + '_' + args.model + '_noniid_unbalanced.out'
    if(os.path.isfile(result_file)):
        os.remove(result_file)
    f_result = open(result_file, 'w')
    run(rank, size, model, args, test_data, f_result)