import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision

from original_model_sq import Net
from evaluate import calculate_topk
from test_model import process_loader
from loader import CustomFolder

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir",default='data',type=str)
parser.add_argument("--model-name",default='ckpt',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument("--lr",default=0.1, type=float)
parser.add_argument("--interval",'-i',default=20,type=int)
parser.add_argument('--resume', '-r',action='store_true')
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

print(f"DEVICE: {device}")
# data loading
root = args.data_dir
model_name = args.model_name
model_path = os.path.join("checkpoint", model_name) + '.t7'
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")
query_dir = os.path.join(root, "query")
gallery_dir = os.path.join(root, "gallery")

# (128,64)

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),  #torchvision.transforms.RandomCrop((335,335), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
    batch_size=64, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
    batch_size=64, shuffle=True
)

# Add query and gallery loader
queryloader = torch.utils.data.DataLoader(
    CustomFolder(query_dir, transform=transform_test),
    batch_size=64, shuffle=False
)
galleryloader = torch.utils.data.DataLoader(
    CustomFolder(gallery_dir, transform=transform_test),
    batch_size=64, shuffle=False
)

num_classes = max(len(trainloader.dataset.classes), len(testloader.dataset.classes))
print(f"Num classes: {num_classes}")
# net definition
start_epoch = 0
net = Net(num_classes=num_classes)
if args.resume:
    assert os.path.isfile(model_path), "Error: no checkpoint file found!"
    print(f'Loading from {model_path}')
    checkpoint = torch.load(model_path)
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
best_acc = 0.
best_epoch = -1

# train function for each epoch
def train(epoch):
    print("\nEpoch : %d"%(epoch))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    print("TRAINING RESULTS")
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
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print 
        if (idx+1)%interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss/interval, correct, total, 100.*correct/total
            ))
            training_loss = 0.
            start = time.time()
    
    return train_loss/len(trainloader), 1.- correct/total, correct/total

def test_topk(epoch):
    global best_acc
    global best_epoch
    net.eval()
    net.reid = True  # Cut the network to get features in forward pass
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        qf, ql = process_loader(queryloader, net, device)
        gf, gl = process_loader(galleryloader, net, device)

        print("--------------------------------------------------------")
        print("Testing TOPK...")
        calculate_topk(qf, ql, gf, gl, 1)
        calculate_topk(qf, ql, gf, gl, 3)

        end = time.time()
        top1, topkcorrect1, total = calculate_topk(qf, ql, gf, gl, 1)
        # top3, topkcorrect3, _ = calculate_topk(qf, ql, gf, gl, 3)
        # top5, topkcorrect5, _ = calculate_topk(qf, ql, gf, gl, 5)
        print("time:{:.2f}s Loss:{:.5f} Correct:{}/{} Top1:{:.3f}%".format(
            end - start, 1-top1, topkcorrect1, total,
            top1
        ))
        print("--------------------------------------------------------")

    net.reid = False  # Return to complete network for forward pass in training


    # saving checkpoint
    acc = top1
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        print(f"Saving parameters to {model_path}")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, model_path)

    return top1, 1. - top1


def test(epoch):
    global best_acc
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
        print(f"Saving parameters {model_path}")
        checkpoint = {
            'net_dict':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, model_path)"""

    return test_loss/len(testloader), 1.- correct/total

# plot figure
x_epoch = []
record = {'train_acc':[], 'train_err':[], 'top1_acc':[], 'top1_err':[]}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="Accuracy")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_acc'].append(train_loss)
    record['train_err'].append(train_err)
    record['top1_acc'].append(test_loss)
    record['top1_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_acc'], 'bo-', label='train')
    ax0.plot(x_epoch, record['top1_acc'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['top1_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()

    if not os.path.isdir('training_results'):
        os.mkdir('training_results')
    fig.savefig(f"training_results/train_{model_name}.jpg")

# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))

def main():

    top1_acc, test_err = test_topk(0)
    for epoch in range(start_epoch, start_epoch+40):
        train_loss, train_err, train_acc = train(epoch)
        #test_loss, test_err = test(epoch)
        top1_acc, test_err = test_topk(epoch)
        draw_curve(epoch, train_acc, train_err, top1_acc, test_err)
        if (epoch+1)%15==0:
            lr_decay()

    global best_epoch
    global best_acc
    print(f"BEST TOP 1: {best_acc} (Epoch = {best_epoch})")


if __name__ == '__main__':
    main()
