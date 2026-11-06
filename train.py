import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models import *


def train_loop(dataloader, model, loss_fn, optimizer, device, inputs, nsgd):
    model.train()
    size = len(dataloader.dataset)
    #pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    train_loss = 0.0
    train_acc = 0.0
    train_acc_top5 = 0.0


    for batch, sample in enumerate(dataloader):

        x,labels = sample
        x,labels = x.to(device), labels.to(device)
        
        outputs = model(x)
        
        for name,module in model.named_modules():
            if name != '' and isinstance(module, (nn.Linear, nn.Conv2d)):
#                if isinstance(module, nn.Linear) and len(inputs[name].size()) == 3:
#                    for param_name, param in module.named_parameters():
#                        if 'bias' not in param_name:
#                            param.input_to_norm = inputs[name].view(inputs[name].size()[0], inputs[name].size()[1] * inputs[name].size()[2])
#                        else:
#                            param.input_to_norm = None
                if isinstance(module, nn.Conv2d):
                    if module.groups != 1:
                        for param_name, param in module.named_parameters():
                            if 'bias' not in param_name:
                                param.input_to_norm = [inputs[name], True]
                            else:
                                param.input_to_norm = None
                    else:
                        for param_name, param in module.named_parameters():
                            if 'bias' not in param_name:
                                param.input_to_norm = [inputs[name], False]
                            else:
                                param.input_to_norm = None
                else:
                    for param_name, param in module.named_parameters():
                        if 'bias' not in param_name:
                            param.input_to_norm = inputs[name]
                        else:
                            param.input_to_norm = None
            else:
                for param_name, param in module.named_parameters():
                    param.input_to_norm = None

        loss = loss_fn(outputs, labels)
        #error = loss_fn.last_error
        #error = torch.sum(error ** 2)



        _,pred = torch.max(outputs,1)
        _,pred2 = torch.topk(outputs,5, dim=1)

        num_correct = (pred == labels).sum()
        top5_correct = torch.eq(pred2, labels.view(-1, 1)).sum()


        if nsgd == 1 or nsgd == 3:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif nsgd == 2:
            optimizer.zero_grad()
            loss.backward()
            def closure():
                optimizer.zero_grad()
                outputs = model(x)
                loss = loss_fn(outputs, labels)
                loss.backward()
                return loss
            optimizer.step(closure)
        
        loss = loss.item()
        acc = num_correct.item()/len(labels)
        count += len(labels)
        train_loss += loss*len(labels)
        train_acc += num_correct.item()
        #pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")

        #top5_acc = top5_correct.item()/len(labels)
        train_acc_top5 += top5_correct.item()
        print(torch.mps.current_allocated_memory())
        
    return train_loss/count, train_acc/count,  train_acc_top5/count