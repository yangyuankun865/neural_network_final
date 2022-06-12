'''Train CIFAR100 with PyTorch.'''
import imp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import random

from models import *
from utils import progress_bar
from torch.autograd import Variable
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
val_percent = 0.2
trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)

n_val = int(len(trainset) * val_percent)
n_train = len(trainset) - n_val
trainset, valset = torch.utils.data.random_split(trainset, [n_train, n_val])

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
# valloader = torch.utils.data.DataLoader(
#     valset, batch_size=256, shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

def kaiming_init(net):
    nn.init.kaiming_normal_(net.conv1.weight)
    for i in range(2):
        nn.init.kaiming_normal_(net.layer1[i].conv1.weight)
        nn.init.kaiming_normal_(net.layer1[i].conv2.weight)
        nn.init.kaiming_normal_(net.layer2[i].conv1.weight)
        nn.init.kaiming_normal_(net.layer2[i].conv2.weight)
        nn.init.kaiming_normal_(net.layer3[i].conv1.weight)
        nn.init.kaiming_normal_(net.layer3[i].conv2.weight)
        nn.init.kaiming_normal_(net.layer4[i].conv1.weight)
        nn.init.kaiming_normal_(net.layer4[i].conv2.weight)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training
writer = SummaryWriter()
def train(net, optimizer, scheduler, criterion, epoch, alpha):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha)
        inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
        # writer.add_scalar('Loss_train',loss, epoch)  # tensorboard train loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def train_val(net, optimizer, scheduler, criterion, epoch, train_loss_list, train_acc, name):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            train_loss_list.append(loss.item())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            train_acc.append(100. * correct / total)
            
            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            writer.add_scalar(str(name)+"/loss_train", loss.item(), epoch)
            writer.add_scalar(str(name)+"/acc_train", 100.*correct/total, epoch)  

def val(net, optimizer, scheduler, criterion, epoch, val_loss_list, val_acc, name):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss_list.append(loss.item())
            

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            val_acc.append(100.*correct/total)

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            writer.add_scalar(str(name)+"/loss_val", loss.item(), epoch) 
            writer.add_scalar(str(name)+"/acc_val", 100.*correct/total, epoch)  




def test(net, optimizer, scheduler, criterion, epoch, best_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    # writer.add_scalar('Acc', acc, epoch)  # tensorboard acc
    if acc > best_acc:
        # print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt10.pth')
        best_acc = acc
    return best_acc
        

def main():
    
    alpha = 0.1
    # alpha_list =  [0.1, 1, 10]
    if not os.path.isdir('./val_loss_list'):
        os.mkdir('val_loss_list')
        os.mkdir('val_acc_list')
        os.mkdir('train_loss_list')
        os.mkdir('train_acc_list')
    for seed in [100,200,300,400,500,600]:
        print("seed",seed)
        setup_seed(seed)
        best_acc = 0
        net = ResNet18().to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        # kaiming_init(net)
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()
        val_loss_list = []
        val_acc = []
        train_loss_list = []
        train_acc = []
        for epoch in range(start_epoch, start_epoch+args.epochs):
            train(net, optimizer, scheduler, criterion, epoch, alpha)
            train_val(net, optimizer, scheduler, criterion, epoch, train_loss_list, train_acc, str(seed))
            val(net, optimizer, scheduler, criterion, epoch, val_loss_list, val_acc, str(seed))
            best_acc = test(net, optimizer, scheduler, criterion, epoch, best_acc)
            scheduler.step()
            if (epoch+1) % 10 == 0:
                print("epoch", epoch, "train_loss", train_loss_list[-1], "val_loss", val_loss_list[-1],
                    "train_acc", train_acc[-1], "val_acc", val_acc[-1])
        print("best_acc", best_acc)
        # torch.save(val_loss_list, os.path.join('val_loss_list',str(alpha)))
        # torch.save(val_acc, os.path.join('val_acc_list', str(alpha)))
        # torch.save(train_loss_list, os.path.join('train_loss_list', str(alpha)))
        # torch.save(train_acc, os.path.join('train_acc_list', str(alpha)))

if __name__=='__main__':
    main()