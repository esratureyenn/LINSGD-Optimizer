import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models import *


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)

    #pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    test_loss = 0.0
    test_acc = 0.0
    test_acc_top5 = 0.0
    with torch.no_grad():
        for batch, sample in enumerate(dataloader):
            x,labels = sample
            x,labels = x.to(device), labels.to(device)

            outputs = model(x)
            loss = loss_fn(outputs, labels)
            _,pred = torch.max(outputs,1)
            _,pred2 = torch.topk(outputs,5, dim=1)
            num_correct = (pred == labels).sum()
            top5_correct = torch.eq(pred2, labels.view(-1, 1)).sum()
            
            loss = loss.item()
            acc = num_correct.item()/len(labels)
            count += len(labels)
            test_loss += loss*len(labels)
            test_acc += num_correct.item()
            test_acc_top5 += top5_correct.item()
            #pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")
    
    return test_loss/count, test_acc/count, test_acc_top5/count