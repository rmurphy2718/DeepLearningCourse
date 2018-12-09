from __future__ import print_function

import os
import sys
import time
import numpy as np
import argparse
import datetime
import math
import pickle
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import bayesian_config as cf

from utils.BBBlayers import GaussianVariationalInference
from utils.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from utils.BayesianModels.Bayesian2Conv3FC import BBB2Conv3FC
from utils.Cross_Validation_Loader import CrossValidationLoader
#from utils.BayesianModels.ByesianAlexNet import BBBAlexNet
#from utils.BayesianModels.BayesianLeNet import BBBLeNet
#from utils.BayesianModels.BayesianSqueezeNet import BBBSqueezeNet

BASE_DIR = '/homes/murph213/DeepLearning/code_final/'

def save_plots(losses, _, train_accs, test_accs, outfile, num_epoch):
    """Plot

        Plot two figures: loss vs. epoch and accuracy vs. epoch
    """
    n = len(losses)
    xs = np.arange(n)

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(xs, losses, '--', linewidth=2, label='loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    log.write("processing data....\n")
    ax.legend(loc='lower right')
    plt.savefig(outfile + "loss.png")

    # plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, train_accs, '--', linewidth=2, label='train')
    ax.plot(xs, test_accs, '-', linewidth=2, label='test')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    plt.savefig(outfile + '_accuracy.png')
    plt.close()


# Return network & file name
def getNetwork(net_type):
    # if args.net_type == 'lenet':
    #     net = BBBLeNet(outputs,inputs)
    #     file_name = 'lenet'
    # elif args.net_type == 'alexnet':
    #     net = BBBAlexNet(outputs,inputs)
    #     file_name = 'alexnet-'
    # elif args.net_type == 'squeezenet':
    #     net = BBBSqueezeNet(outputs,inputs)
    #     file_name = 'squeezenet-'
    if net_type == '3conv3fc':
        net = BBB3Conv3FC(outputs, inputs)
        file_name = '3Conv3FC-'
    elif net_type == '2conv3fc':
        net = BBB2Conv3FC(outputs, inputs)
        file_name = '2Conv3FC-'
    else:
        print('Error : Network not recognized')
        sys.exit(0)

    model_parameters = filter(lambda ppp: ppp.requires_grad, net.parameters())
    num_params = sum([np.prod(ppp.size()) for ppp in model_parameters])

    print(num_params)

    return net, file_name


def train(args, epoch, net, trainloader, vi, train_size, learning_rate, batch_size):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    log.write('\n=> Training Epoch %d, LR=%.4f\n' %(epoch, cf.learning_rate(learning_rate, epoch)))
    for batch_idx, (inputs_value, targets) in enumerate(trainloader):
        x = inputs_value.view(-1, inputs, resize, resize).repeat(args.num_samples, 1, 1, 1)
        y = targets.repeat(args.num_samples)
        if use_cuda:
            x, y = x.cuda(), y.cuda() # GPU settings

        # log.write(str(x.shape) + "\n")
        # log.write(str(y.shape) + "\n")

        # Forward Propagation
        x, y = Variable(x), Variable(y)
        outputs, kl = net.probforward(x)

        beta = 1.0 / float(batch_size)
        loss = vi(outputs, y, kl, beta)  # Loss

        optimizer.zero_grad()
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

        log.write('\r')
        log.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Accuracy: %.3f%% \n'
                %(epoch, num_epochs, batch_idx+1,
                    (train_size//batch_size)+1, loss.data.item(), (100*correct/total)/args.num_samples))

    acc = (100 * correct / total) / args.num_samples

    diagnostics_to_write =  {'Epoch': epoch, 'Loss': loss.item(), 'Accuracy': acc.item()}
    log.write(str(diagnostics_to_write) + "\n")
    log.flush()

    return loss, acc

def test(args, epoch, net, testloader, vi, testsize, batch_size):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        x = inputs_value.view(-1, inputs, resize, resize).repeat(args.num_samples, 1, 1, 1)
        y = targets.repeat(args.num_samples)
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            x, y = Variable(x), Variable(y)

        outputs, kl = net.probforward(x)

        beta = 1 / batch_size

        loss = vi(outputs, y, kl, beta)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

    # Save checkpoint when best model
    # print(correct)
    # print(total)
    acc = (100.0*float(correct)/float(total))/float(args.num_samples)
    log.write("\n| Validation Epoch #%d\n \t\t\tLoss: %.4f Accuracy: %.2f%%" %(epoch, loss.data.item(), acc))
    log.write("\n")
    test_diagnostics_to_write = {'Validation Epoch': epoch, 'Loss':loss.item(), 'Accuracy': acc}
    log.write(str(test_diagnostics_to_write))
    log.write("\n")
    #
    # if acc > best_acc:
    #     log.write('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
    #     state = {
    #             'net':net if use_cuda else net,
    #             'acc':acc,
    #             'epoch':epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     save_point = './checkpoint/'+args.dataset+os.sep
    #     if not os.path.isdir(save_point):
    #         os.mkdir(save_point)
    #     torch.save(state, save_point+file_name+'.t7')
    #     best_acc = acc

    return loss, acc


def fit(args, train_loader, test_loader, suffix, log, num_epochs, train_size, test_size, learning_rate, batch_size):
    log.write('\n[Phase 2] : Model setup\n')
    log.write('| Building net type [' + args.net_type + ']...\n')

    net, file_name = getNetwork(args.net_type)

    if use_cuda:
        net.cuda()

    vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())

    elapsed_time = 0
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train(args, epoch, net, train_loader, vi, train_size, learning_rate, batch_size)
        val_loss, val_acc = test(args, epoch, net, test_loader, vi, test_size, batch_size)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        log.write('\n| Elapsed time : %d:%02d:%02d  \n'  %(cf.get_hms(elapsed_time)))


        outpath = os.path.join(BASE_DIR, 'weights', 'w_{}'.format(suffix) + ".pt")
        torch.save(net.state_dict(), outpath)

    plot_file = os.path.join(BASE_DIR, "output", suffix)
    save_plots(train_losses, val_losses, train_accs, val_accs, plot_file, num_epochs)


# ---------------------------------------------------
#
# Parse arguments
#
# ---------------------------------------------------

# Note: hyper_and_out dictionary is defined in the config file
# This maps positive integers to hyperparm settings
# with learning rate and batch size
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
#parser.add_argument('--lr', default=0.0001, type=float, help='learning_rate')
parser.add_argument('--net_type', default='3conv3fc', type=str, help='model')
parser.add_argument('--num_samples', default=10, type=int, help='Number of samples')
parser.add_argument('--beta_type', default="Standard", type=str, help='Beta type')
parser.add_argument('--p_logvar_init', default=0, type=int, help='p_logvar_init')
parser.add_argument('--q_logvar_init', default=-10, type=int, help='q_logvar_init')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight_decay')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset = [mnist/cifar10/cifar100/fashionmnist/stl10]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--config_integer', type=int, help='Which hyperparm configuration setting')
parser.add_argument('--num_folds', type=int, help='Number of folds in k-fold cross validation')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
# parser.add_argument('--batch_size', type=int, help='Minibatch size')
args = parser.parse_args()

assert args.dataset == "mnist", "Assuming MNIST for now."
assert args.net_type in ("3conv3fc", "2conv3fc"), "Invalid network type"

if args.net_type == "3conv3fc":
    print("Using the FULL architecture")
else:
    print("Using the PRUNED architecture")

# Give time to quit if user entered wrong architecture
time.sleep(5)

# ---------------------------------------------------
# Set GPU stuff
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)


# Hyper and Global Parameter settings
best_acc = 0
resize = 32
num_epochs = args.num_epochs
optim_type = 'Adam'
# Load the hyperparm dic.  It's in output b/c we'll put output stats in it.
hyper_and_out = pickle.load(open(os.path.join(BASE_DIR, 'output', 'hyper_and_out.pkl'), 'rb'))
batch_size = hyper_and_out[args.config_integer]['batch_size']
learning_rate = hyper_and_out[args.config_integer]['lr']
outputs = 10
inputs = 1

# ----------------------------------------------------------
# Data Load
# Here we set up k-fold cross validation.
# ----------------------------------------------------------
print("Assuming MNIST")

transform_train = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])  # meanstd transformation

transform_test = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

# ASSUMING MNIST
#
# if (args.dataset == 'cifar10'):
#     print("| Preparing CIFAR-10 dataset...")
#     sys.stdout.write("| ")
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#     testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
#     outputs = 10
#     inputs = 3
#
# elif (args.dataset == 'cifar100'):
#     print("| Preparing CIFAR-100 dataset...")
#     sys.stdout.write("| ")
#     trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
#     testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
#     outputs = 100
#     inputs = 3
#
# elif (args.dataset == 'mnist'):
#     print("| Preparing MNIST dataset...")
#     sys.stdout.write("| ")
#     trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
#     testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
#     outputs = 10
#     inputs = 1
#
# elif (args.dataset == 'fashionmnist'):
#     print("| Preparing FASHIONMNIST dataset...")
#     sys.stdout.write("| ")
#     trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
#     testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_test)
#     outputs = 10
#     inputs = 1
# elif (args.dataset == 'stl10'):
#     print("| Preparing STL10 dataset...")
#     sys.stdout.write("| ")
#     trainset = torchvision.datasets.STL10(root='./data',  split='train', download=True, transform=transform_train)
#     testset = torchvision.datasets.STL10(root='./data',  split='test', download=False, transform=transform_test)
#     outputs = 10
#     inputs = 3

# ----------------------------------------------------------
# Create a cross validation loader
# Returns a 'training' loader with (k-1) folds and a 'test' loader with last fold
# ----------------------------------------------------------
cross_valid_loader = CrossValidationLoader(transform_train,
                                           outputs,
                                           inputs,
                                           batch_size,
                                           args.num_folds)
iterator = iter(cross_valid_loader)


for ii in range(len(iterator)):
    # Files get similar names but are put in different directories
    # Directory is prefix, name is suffix
    suffix = 'setting_{}_fold_{}'.format(args.config_integer, ii)
    # prepend indication of pruned model
    if args.net_type == '2conv3fc':
        suffix = "pruned_" + suffix

    logfile = os.path.join(BASE_DIR, 'logs', suffix + ".txt")
    log = open(logfile, 'w+')

    log.write("\n")
    log.write("="*20)
    log.write("\nFold {}".format(ii))
    log.write("\n")
    log.write("="*20)
    log.write('\n| Training Epochs = ' + str(num_epochs) + "\n")
    log.write('| Initial Learning Rate = ' + str(learning_rate) + "\n")
    log.write('| Optimizer = ' + str(optim_type) + "\n")

    log.write("processing data....\n")
    train_loader, test_loader, train_size, test_size = next(iterator)
    log.write("done\n")
    log.flush()

    fit(args, train_loader, test_loader, suffix, log,
        num_epochs, train_size, test_size, learning_rate, batch_size)

    log.write("\n")
    log.close()

print("finished")

