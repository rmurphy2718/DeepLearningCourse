"""
do_pruning
Ryan Murphy, Jincheng Bai
"""

from utils.BBBlayers import GaussianVariationalInference
from utils.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from utils.BayesianModels.BayesianAlexNet import BBBAlexNet
from utils.BayesianModels.BayesianLeNet import BBBLeNet
from utils.BayesianModels.BayesianSqueezeNet import BBBSqueezeNet

from utils_new import *

import torch
import numpy as np
import argparse
import matplotlib
import torchvision
import torchvision.transforms as transforms
import bayesian_config as cf
import sys
import math
from torch.autograd import Variable

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test(net, args):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    m = math.ceil(len(testset) / batch_size)
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        print('testing..')

        x = inputs_value.view(-1, inputs, resize, resize).repeat(2, 1, 1, 1)
        y = targets.repeat(2)
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            x, y = Variable(x), Variable(y)
        outputs, kl = net.probforward(x)


        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

    acc = (100 * correct / total) / 2.0

    return acc


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Training for MNIST')
    # parser.add_argument('data_folder', metavar='DATA_FOLDER',
    #                     help='the folder that contains all the input data')

    parser.add_argument('-id', '--indir')
    parser.add_argument('-if', '--infile')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')

    parser.add_argument('-ns', '--num_samples', type=int,
                        help='show info messages')

    parser.add_argument('-ds', '--dataset', default='mnist')

    args = parser.parse_args(argv)
    return args


def sort_by_col(arr, col):
    """sort numpy array by column col
    :param col
    """
    return arr[arr[:, col].argsort()]


def get_sig_to_noise(mean_vec, logvar_vec):
    """

    :param mean_vec: torch tensor of means of each weight in a filter
    :param logvar_vec:
    :return: signal to noise ratio as a float
    """
    if len(mean_vec.shape) > 1:
        mean_vec = mean_vec.view(-1)

    if len(logvar_vec.shape) > 1:
        logvar_vec = logvar_vec.view(-1)

    var_vec = torch.exp(logvar_vec)
    return torch.mean((mean_vec**2)/var_vec**2).item()


def get_sorted_stn_idx(filter_means, filter_logvars):
    """
    Get the signal-to-noise ratio for each filter
    :param: Torch parameters, listing all filter mean & logvar for all filters
    :return: A list of indices: first element is lowest sig to noise, last is highest
    """
    num_filts = filter_means.shape[0]
    idx_by_stn = np.empty((num_filts, 2))
    for filt_idx in range(num_filts):
        idx_by_stn[filt_idx, 0] = filt_idx
        idx_by_stn[filt_idx, 1] = get_sig_to_noise(filter_means[filt_idx], filter_logvars[filt_idx])

    return sort_by_col(idx_by_stn, 1)


def save_hist(arr, outfile, conv_layer):
    print(outfile)
    """Make histogram of numpy array and save to disk"""
    plt.hist(arr, bins='auto')
    plt.title("Signal to noise distribution Conv Layer {}".format(conv_layer))
    plt.savefig(outfile)
    plt.close()

if __name__ == "__main__":
    args = bin_config(get_arguments)
    batch_size = cf.batch_size
    resize = 32
    transform_test = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean['mnist'], cf.std[args.dataset]),
    ])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #
    # load model
    #
    outputs = 10
    inputs = 1

    model = BBB3Conv3FC(outputs, inputs)
    #
    path = args.indir + "/" + args.infile
    model.load_state_dict(torch.load(path))
    model.to(device)
    #
    # load data
    #
    print("| Preparing MNIST dataset...")
    sys.stdout.write("| ")
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
    outputs = 10
    inputs = 1
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    #
    # Test
    #
    accuracy = test(model, args)
    print("="*20)
    print("The acc is {}".format(accuracy))
    print("="*20)
    #
    # Get signal to noise distribution for all conv layers
    #
    stn_1 = get_sorted_stn_idx(model.conv1.qw_mean, model.conv1.qw_logvar)
    stn_2 = get_sorted_stn_idx(model.conv2.qw_mean, model.conv2.qw_logvar)
    stn_3 = get_sorted_stn_idx(model.conv3.qw_mean, model.conv3.qw_logvar)

    print(model.conv3.qw_mean.shape)
    #
    # Save histogram to disk
    #
    fname = "histogram_{}".format(args.infile)
    save_hist(stn_1[:, 1], args.indir + "/" + fname + "1.png", 1)
    save_hist(stn_2[:, 1], args.indir + "/" + fname + "2.png", 2)
    save_hist(stn_3[:, 1], args.indir + "/" + fname + "3.png", 3)
    #
    # Drop the last layer
    #

