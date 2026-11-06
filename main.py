import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import warnings
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from read_data import read_cifar10, read_cifar100
#from models import *
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from train import train_loop
from test import test_loop

from sgd import SGD
from NISGD_loss import NISGD_loss
from NISGD_Exp import NISGD_Exp
from cross_entropy_loss_with_error import CrossEntropyWithError
from functions import save_args

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-m', '--model', default='wide_resnet18_flat', type=str,
                    dest='model',  help='Which model to use. The option to choose are resnet_flat, resnet18, resnet50, customcnn, vit, wide_resnet18_flat')
parser.add_argument('-lr', '--learning_rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--norm_t', default=2, type=int, help='norm_t')
parser.add_argument('--loss_weight', default=0.3, type=int, help='loss_weight * loss + (1 - loss_weight) * norm_input')
parser.add_argument('--dataset', default=1, type=int, help='0 for cifar10, 1 for cifar100')
parser.add_argument('--main_dir', type=str, default='./Results_cifar100')
parser.add_argument('--nsgd', type=int, default=3, dest='nsgd',
                    help='Which optimizer to use. 1 for sgd, 2 for loss normalized insgd, 3 for insgd')
parser.add_argument('--device', type=str, default='cuda', help='It can be mps(macbook), cuda(windows,colab) and cpu')


def main():
    
    args = parser.parse_args()    
    


    #seed_list = 0

    insgd_list = [1, 2, 3]
    norm_t_list = [1, 2]
    lr_list = [0.1, 0.03, 0.01, 0.003]
    batch_list = [64, 128]
    loss_weight_list = [0.3, 0.5]



    for insgd_step in range(len(insgd_list)):
        for norm_step in range(len(norm_t_list)):
            for loss_step in range(len(loss_weight_list)):
                for lr_step in range(len(lr_list)):
                    for batch_step in range(len(batch_list)):
                        nsgd = insgd_list[insgd_step]
                        norm_t = norm_t_list[norm_step]
                        loss_weight = loss_weight_list[loss_step]
                        learning_rate = lr_list[lr_step]
                        batch_size = batch_list[batch_step]


                        args.nsgd = nsgd
                        args.lr = learning_rate
                        args.loss_weight = loss_weight
                        args.norm_t = norm_t
                        args.batch_size = batch_size
                        
                        if nsgd == 1:
                            pass
                        elif  nsgd == 2 and norm_t == 1:
                            pass
                        elif  nsgd == 2 and norm_t ==  2 and loss_weight == 0.3:
                            pass
                        elif  nsgd == 2 and norm_t ==  2 and loss_weight == 0.5 and learning_rate == 0.1:
                            pass
                        else:
                            if nsgd == 2:
                                main_worker(args)
                            elif nsgd == 3: 
                                if loss_weight == 0.3:
                                    main_worker(args)
                                else:
                                    pass


         






def  main_worker(args):


    seed = args.seed
    device = args.device
    main_dir = args.main_dir
    batch_size = args.batch_size
    epochs = args.epochs
    weight_decay  = args.weight_decay
    momentum = args.momentum
    model_type = args.model
    nsgd = args.nsgd
    lr = args.lr
    dataset = args.dataset
    norm_t = args.norm_t
    loss_weight = args.loss_weight

    def my_seed_every(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)



    if nsgd == 1:
        main_dir_alt = "sgd"
    elif nsgd == 2:
        main_dir_alt = "insgd_loss"
    elif nsgd == 3:
        main_dir_alt = "insgd"

    save_dir = "norm_t_%d_loss_weight_%.4f_learning_rate_%.4f_batch_size_%d" % (norm_t, loss_weight, lr, batch_size)

    # Directory creation
    main_dir2 = os.path.join(main_dir, main_dir_alt)
    directory = os.path.join(main_dir2, save_dir)

    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    if not os.path.exists(main_dir2):
        os.mkdir(main_dir2)
    if not os.path.exists(directory):
        os.mkdir(directory)


    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    data_dir = 'data'


    # CIFAR-10 dataloader generator

    if dataset == 0:
        train_generator,test_generator = read_cifar10(batch_size,data_dir, my_seed_every, seed, seed_worker, g, False)
    elif dataset == 1:
        train_generator,test_generator = read_cifar100(batch_size,data_dir, my_seed_every, seed, seed_worker, g, False)

    # Model type
    my_seed_every(seed=seed)
    if model_type == 'resnet18':
        from models.resnet import resnet18
        model = resnet18().to(device)
    elif model_type == 'resnet50':
        from models.resnet import resnet50
        model = resnet50().to(device)
    elif model_type == 'resnet_flat':
        from models.resnet_flat import ResNet20_Flat
        model = ResNet20_Flat().to(device)
    elif model_type == 'customcnn':
        from models.customcnn import SimpleModel_Cifar
        model = SimpleModel_Cifar().to(device)
    elif model_type == 'vit':
        from models.vit import ViT
        model = ViT(image_size=32, patch_size=4, num_classes=10, dim=512, depth=4, heads=6, mlp_dim=512).to(device)
    elif model_type == 'wide_resnet18_flat':
        from models.resnet_flat import Wide_ResNet18_Flat
        model = Wide_ResNet18_Flat().to(device)


    print(device)
    print("There are", sum(p.numel() for p in model.parameters()), "parameters.")
    print("There are", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")



    inputs = {}



    def get_inputs(name):
        def hook(model, input, output):
            inputs[name] = input[0].detach()
        return hook
        
    for name,module in model.named_modules():
        for param_name, param in module.named_parameters():
            param.input_to_norm = None
        if name != '' and isinstance(module, (nn.Linear, nn.Conv2d)):
            module.register_forward_hook(get_inputs(name))




    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    #criterion = CrossEntropyWithError().to(device)
    #criterion = MyMSELoss().to(device)
    #criterion = Error_Norm_MyMSELoss().to(device)

    if nsgd == 1:
        print('Optimizer is SGD from scratch')
        optimizer = SGD(model.parameters(),lr=lr,
                                     momentum=momentum,
                                     weight_decay=weight_decay)


    elif nsgd == 2:
        print('Optimizer is loss normalized INSGD')
        optimizer = NISGD_loss(model.parameters(), lr=lr,
                              momentum=momentum,
                              weight_decay=weight_decay,
                              loss_weight=loss_weight,
                              norm_t=norm_t) 
  
  
    elif nsgd == 3:
        print('Optimizer is INSGD')
        optimizer = NISGD_Exp(model.parameters(), lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            norm_t=norm_t)
        
        
    #metric = MulticlassAccuracy(num_classes=num_classes).to(device)


    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=args.gamma,last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    idx_best_loss = 0
    idx_best_acc = 0
    idx_best_acc_top5 = 0
    
    list_train_loss = []
    list_train_acc = []
    list_train_acc_top5 = []
    list_test_loss = []
    list_test_acc = []
    list_test_acc_top5 = []

    all_results = []


    training_file = os.path.join(directory, 'training.txt')

    for epoch in range(1, epochs+1):
        with open(training_file, 'a') as f:
            print("epoch:",epoch)
            f.write("\n")
            f.write("===> Epoch {}/{}".format(epoch, epochs))
        #print("===> Epoch {}/{}".format(epoch, n_epochs))
        start_time = time.time()
        train_loss, train_acc, train_acc_top5 = train_loop(train_generator, model, criterion, optimizer, device, inputs, nsgd)
        test_loss, test_acc, test_acc_top5 = test_loop(test_generator, model, criterion, device)
        time_spent = time.time() - start_time

        if dataset == 0 or dataset == 2:
            with open(training_file, 'a') as f:
                f.write("\n")
                f.write("Time spent:{:f}".format(time_spent))
                f.write("\n")
                f.write("Training loss: {:f}, acc: {:f}".format(train_loss, train_acc))
                f.write("\n")
                f.write("Test loss: {:f}, acc: {:f}".format(test_loss, test_acc))
        elif dataset == 1 or dataset == 3:
            with open(training_file, 'a') as f:
                f.write("\n")
                f.write("Time spent:{:f}".format(time_spent))
                f.write("\n")
                f.write("Training loss: {:f}, acc: {:f}, top5 acc: {:f}".format(train_loss, train_acc, train_acc_top5))
                f.write("\n")
                f.write("Test loss: {:f}, acc: {:f}, top5 acc: {:f}".format(test_loss, test_acc, test_acc_top5))
        #print("Time spent:{:f}".format(time_spent))
        #print("Training loss: {:f}, acc: {:f}".format(train_loss, train_acc))
        #print("Test loss: {:f}, acc: {:f}".format(test_loss, test_acc))

        scheduler.step()
        
        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        list_train_acc_top5.append(train_acc_top5)
        list_test_loss.append(test_loss)
        list_test_acc.append(test_acc)
        list_test_acc_top5.append(test_acc_top5)
        
        if test_loss <= list_test_loss[idx_best_loss]:
            #print("Loss-best model.")
            torch.save(model.state_dict(), os.path.join(directory, 'loss_model.pth'))
            idx_best_loss = epoch - 1
        
        if test_acc >= list_test_acc[idx_best_acc]:
            #print("Acc-best model.")
            torch.save(model.state_dict(), os.path.join(directory, 'acc_model.pth'))
            if test_acc > list_test_acc[idx_best_acc]:
                idx_best_acc = epoch - 1    

        if test_acc_top5 >= list_test_acc_top5[idx_best_acc_top5]:
            #print("Acc-best model.")
            torch.save(model.state_dict(), os.path.join(directory, 'acc_top5_model.pth'))
            if test_acc_top5 > list_test_acc_top5[idx_best_acc_top5]:
                idx_best_acc_top5 = epoch - 1    
        #print("")

    if dataset == 0 or dataset == 2:
        with open(training_file, 'a') as f:
            f.write("\n")
            f.write("========================")
            f.write("\n")
            f.write("Loss-best model training loss: {:f}, acc: {:f}".format(list_train_loss[idx_best_loss], list_train_acc[idx_best_loss]))
            f.write("\n")
            f.write("Loss-best model test loss: {:f}, acc: {:f}".format(list_test_loss[idx_best_loss], list_test_acc[idx_best_loss]))             
            f.write("\n")   
            f.write("Acc-best model training loss: {:4f}, acc: {:f}".format(list_train_loss[idx_best_acc], list_train_acc[idx_best_acc]))  
            f.write("\n")
            f.write("Acc-best model test loss: {:f}, acc: {:f}".format(list_test_loss[idx_best_acc], list_test_acc[idx_best_acc]))    
            f.write("\n")          
            f.write("Final model training loss: {:f}, acc: {:f}".format(list_train_loss[-1], list_train_acc[-1]))     
            f.write("\n")            
            f.write("Final model test loss: {:f}, acc: {:f}".format(list_test_loss[-1], list_test_acc[-1]))  
            f.write("\n")    
            f.write("Test accuracy list:" + str(np.array(list_test_acc)))
    elif dataset == 1 or dataset == 3:
        with open(training_file, 'a') as f:
            f.write("\n")
            f.write("========================")
            f.write("\n")
            f.write("Loss-best model training loss: {:f}, acc: {:f}, top5 acc: {:f}".format(list_train_loss[idx_best_loss], list_train_acc[idx_best_loss], list_train_acc_top5[idx_best_loss]))
            f.write("\n")
            f.write("Loss-best model test loss: {:f}, acc: {:f}, top5 acc: {:f}".format(list_test_loss[idx_best_loss], list_test_acc[idx_best_loss], list_test_acc_top5[idx_best_loss]))             
            f.write("\n")   
            f.write("Acc-best model training loss: {:4f}, acc: {:f}, top5 acc: {:f}".format(list_train_loss[idx_best_acc], list_train_acc[idx_best_acc],list_train_acc_top5[idx_best_acc]))  
            f.write("\n")
            f.write("Acc-best model test loss: {:f}, acc: {:f}, top5 acc: {:f}".format(list_test_loss[idx_best_acc], list_test_acc[idx_best_acc], list_test_acc_top5[idx_best_acc]))    
            f.write("\n")     
            f.write("Acc-top5-best model training loss: {:4f}, acc: {:f}, top5 acc: {:f}".format(list_train_loss[idx_best_acc_top5], list_train_acc[idx_best_acc_top5],list_train_acc_top5[idx_best_acc_top5]))  
            f.write("\n")
            f.write("Acc-top5-best model test loss: {:f}, acc: {:f}, top5 acc: {:f}".format(list_test_loss[idx_best_acc_top5], list_test_acc[idx_best_acc_top5], list_test_acc_top5[idx_best_acc_top5]))    
            f.write("\n")
            f.write("Final model training loss: {:f}, acc: {:f}, top5 acc: {:f}".format(list_train_loss[-1], list_train_acc[-1], list_train_acc_top5[-1]))     
            f.write("\n")            
            f.write("Final model test loss: {:f}, acc: {:f}, top5 acc: {:f}".format(list_test_loss[-1], list_test_acc[-1], list_test_acc_top5[-1]))  
            f.write("\n")    
            f.write("Test accuracy list:" + str(np.array(list_test_acc)))
            f.write("\n")    
            f.write("Test top 5 accuracy list:" + str(np.array(list_test_acc_top5)))
    """
    print("=============================================================")

    print("Loss-best model training loss: {:f}, acc: {:f}".format(list_train_loss[idx_best_loss], list_train_acc[idx_best_loss]))   
    print("Loss-best model test loss: {:f}, acc: {:f}".format(list_test_loss[idx_best_loss], list_test_acc[idx_best_loss]))                
    print("Acc-best model training loss: {:4f}, acc: {:f}".format(list_train_loss[idx_best_acc], list_train_acc[idx_best_acc]))  
    print("Acc-best model test loss: {:f}, acc: {:f}".format(list_test_loss[idx_best_acc], list_test_acc[idx_best_acc]))              
    print("Final model training loss: {:f}, acc: {:f}".format(list_train_loss[-1], list_train_acc[-1]))                 
    print("Final model test loss: {:f}, acc: {:f}".format(list_test_loss[-1], list_test_acc[-1]))           
    #torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
    """
    
    list_train_loss = np.array(list_train_loss)
    list_train_acc = np.array(list_train_acc)
    list_train_acc_top5 = np.array(list_train_acc_top5)
    list_test_loss = np.array(list_test_loss)
    list_test_acc = np.array(list_test_acc)
    list_test_acc_top5 = np.array(list_test_acc_top5)

    all_results.append((list_train_loss, list_train_acc, list_train_acc_top5, list_test_loss, list_test_acc, list_test_acc_top5))

    """
    torch.save(list_train_loss, os.path.join(directory , 'loss_train.pt'))
    torch.save(list_train_acc , os.path.join(directory , 'acc_train.pt'))
    torch.save(list_test_loss, os.path.join(directory , 'loss_test.pt'))
    torch.save(list_test_acc, os.path.join(directory , 'acc_test.pt'))
    """
    torch.save(all_results, os.path.join(directory , 'all_results.pt'))
    torch.save(model.state_dict(), os.path.join(directory, 'model_state_dict.pth'))




    save_args(args, directory)




if __name__ == '__main__':
    main()
