import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision

from utils import calculate_mean_std_per_channel, get_train_valid_loader, test_loader
from original_model import Net


parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir",default='Market-1501-v15.09.15',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--mean-std", action="store_true")
parser.add_argument("--split-valid", action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument("--lr",default=0.1, type=float)
parser.add_argument("--interval",'-i',default=20,type=int)
parser.add_argument('--resume', '-r',action='store_true')
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"

if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loading
root = args.data_dir
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")


# net definition
start_epoch = 0

# loss and optimizer

best_acc = 0.


# train function for each epoch
def train(net, trainloader, criterion, epoch):
    print("\nEpoch : %d"%(epoch+1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    """triplet_training_loss = 0.
    triplet_train_loss = 0."""
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        """triplet_training_loss += triplet_loss.item()
        triplet_train_loss += triplet_loss.item()"""
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print 
        if (idx+1)%interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Cross Entropy Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss/interval, correct, total, 100.*correct/total
            ))
            training_loss = 0.
            start = time.time()
    
    return train_loss/len(trainloader), 1.- correct/total


def validation(net, validloader, criterion, epoch):
    global best_acc
    net.eval()
    valid_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(validloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

        print("Validating ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            100. * (idx + 1) / len(validloader), end - start, valid_loss / len(validloader), correct, total,
            100. * correct / total
        ))

    # saving checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')

    return valid_loss / len(validloader), 1. - correct / total

def test(net, testloader, criterion):
    #global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)
        
        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(testloader), end-start, test_loss/len(testloader), correct, total, 100.*correct/total
            ))

    """# saving checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')"""

    return test_loss/len(testloader), 1. - correct/total

# plot figure
x_epoch = []
record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")

# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))

def create_net(num_classes):
    net = Net(num_classes=num_classes)

    if args.resume:
        assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
        print('Loading from checkpoint/ckpt.t7')
        checkpoint = torch.load("./checkpoint/ckpt.t7")
        # import ipdb; ipdb.set_trace()
        net_dict = checkpoint['net_dict']
        net.load_state_dict(net_dict)
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    net.to(device)

    return net

def main():
    if args.mean_std:
        normalize = calculate_mean_std_per_channel(train_dir, 1)
    else:
        normalize = None

    trainloader, validloader = get_train_valid_loader(
        train_dir,
        batch_size=128,
        augment=False,
        random_seed=1234,
        valid_size=0.05,
        shuffle=True,
        normalize=normalize
    )
    num_classes = len(trainloader.dataset.classes)

    testloader = test_loader(test_dir, batch_size=1, normalize=normalize)

    # Net definition
    net = create_net(num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    #second_criterion = torch.nn.TripletMarginLoss()

    global optimizer
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)


    for epoch in range(start_epoch, start_epoch+40):
        train_loss, train_err = train(net, trainloader, criterion, epoch)
        valid_loss, valid_err = validation(net, validloader, criterion, epoch)
        test_loss, test_err = test(net, testloader, criterion)
        draw_curve(epoch, train_loss, train_err, valid_loss, valid_err)
        if (epoch+1)%20==0:
            lr_decay()


if __name__ == '__main__':
    main()
