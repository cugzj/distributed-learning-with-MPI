import torch
from torch.optim import SGD
import torch.distributed as dist
from torch.autograd import Variable

def update_model(model, weight, group, cpu, gpu):
    all_param = model.state_dict()
    
    # send parameters to PS
    for param in all_param.values():
        param_cpu = torch.tensor(param.data*weight, device=cpu)
        dist.gather(tensor=param_cpu, dst=0, group=group)
    
    # receive parameters from PS
    for key in all_param.keys():
        recv = torch.zeros_like(all_param[key].data, device=cpu)
        dist.scatter(tensor=recv, src=0, group=group)
        all_param[key] = torch.tensor(recv.data, device=gpu)

    model.load_state_dict(all_param)

def run(rank, size, model, args, data_ratio_pairs, cpu, gpu):
    # Send the weights to server 
    weights = [w for _, w in data_ratio_pairs]
    dist.gather(tensor=torch.tensor(weights), dst=0)

    # model = model.cuda(gpu)
    # criterion = torch.nn.CrossEntropyLoss()
    # iterator = iter(train_data)
    # optimizer = SGD(model.parameters(), lr=args.lr)

    # workers = [v+1 for v in range(size-1)]
    # _group = [w for w in workers].append(rank)
    # group = dist.new_group(_group)

    # # Receive initial model from server
    # for idx, p in enumerate(model.parameters()):
    #     tmp_p = torch.zeros_like(p, device=cpu)
    #     dist.scatter(tensor=tmp_p, src=0, group=group)
    #     p.data = torch.tensor(tmp_p, device=gpu)

    # print('Rank {} successfully received the model. '.format(rank))

    # for t in range(args.T):
    #     tot_loss = 0.0
        
    #     # local update 
    #     for k in range(args.E):
    #         try:
    #             data, target = next(iterator)
    #         except:
    #             iterator = iter(train_data)
    #             data, target = next(iterator)
    #         data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, target)
    #         tot_loss = tot_loss + loss.data
    #         loss.backward()
    #         optimizer.step()
        
    #     loss = tot_loss / args.E
    #     print('Rank: {} \tEpoch: {}\t\tLoss: {}'.format(rank, t, loss))

    #     # synchronization 
    #     # send train_loss to PS 
    #     loss_cpu = torch.tensor(loss*weight, device=cpu)
    #     dist.gather(tensor=loss_cpu, dst=0, group=group)
        
    #     # Update the model parameters 
    #     update_model(model, weight, group, cpu, gpu)

    #     if (t+1)%20 == 0:
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] *= 0.95
        

def init_processes(rank, size, model, args, data_ratio_pairs, cpu, gpu, backend='mpi'):
    dist.init_process_group(backend, rank=rank, world_size=size)
    run(rank, size, model, args, data_ratio_pairs, cpu, gpu)