"""
Course project for deep learning, Ryan Murphy and Jincheng Bai

Load the trained models, compute, for each of {full, pruned}
(1) Number of parameters
(2) Histogram of signal to noise ratios (STN)
(3) sample mean and variance of STNs
(4) Test set accuracy
(5) Test set computation time


FINALLY, load them into output/hyper_and_out.pkl
"""

from BBB.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from BBB.BayesianModels.Bayesian2Conv3FC import BBB2Conv3FC

import torch
import argparse
import matplotlib
import torchvision
import torchvision.transforms as transforms
import bayesian_config as cf
from torch.autograd import Variable
import sys
import os
import pickle
import numpy as np
import time

import matplotlib.pyplot as plt
matplotlib.use('Agg')

BASE_DIR = '/homes/murph213/DeepLearning/code_final'


def test(net):
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        #print('testing..')

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

    print("Assuming that two samples were used!")
    acc = (100 * correct / total) / 2.0

    return acc


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
    """Make histogram of numpy array and save to disk"""
    print(outfile)
    plt.hist(arr, bins='auto', facecolor = 'blue', normed=1)
    #plt.title("Signal to noise distribution Conv Layer {}".format(conv_layer))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Proportion")
    plt.savefig(outfile)
    plt.close()
#
# Parse arguments
#
parser = argparse.ArgumentParser(description='Training for MNIST')
parser.add_argument('--config_integer', type=int, help='Hyperparameter configuration integer')
parser.add_argument('--net_type', help="full | pruned")
parser.add_argument('--fold', type=int, help="The cross-val code you want to analyze")
args = parser.parse_args()

assert args.net_type in ('full', 'pruned'), "Network type should be 'full' or 'pruned'"
net_type = args.net_type
config_integer = args.config_integer
fold = args.fold
#
# Set up GPU
#
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#
# Load configuration dictionary
#
output_dir = os.path.join(BASE_DIR, 'output')
config_file = os.path.join(output_dir, '')
hyper_and_out = pickle.load(open(output_dir + '/hyper_and_out.pkl', 'rb'))
batch_size = hyper_and_out[config_integer]['batch_size']
learning_rate = hyper_and_out[config_integer]['lr']
#
# load model
#
outputs = 10
inputs = 1
if net_type == 'full':
    model = BBB3Conv3FC(outputs, inputs)
    train_weights_path = os.path.join(BASE_DIR, 'weights', 'w_setting_{}_fold_{}.pt'.format(config_integer, fold))
else:
    model = BBB2Conv3FC(outputs, inputs)
    train_weights_path = os.path.join(BASE_DIR, 'weights', 'w_pruned_setting_{}_fold_{}.pt'.format(config_integer, fold))

model.load_state_dict(torch.load(train_weights_path))
#
# (1)  Count the number of params
#
model_parameters = filter(lambda ppp: ppp.requires_grad, model.parameters())
num_params = sum([np.prod(ppp.size()) for ppp in model_parameters])

print("Number params {}".format(num_params))
model.to(device)
#
# Load Data
#
resize = 32
transform_test = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['mnist'], cf.std['mnist']),
])

print("| Preparing MNIST dataset...")
sys.stdout.write("| ")

testset = torchvision.datasets.MNIST(root=os.path.join(BASE_DIR, 'data'),
                                     train=False,
                                     download=False,
                                     transform=transform_test)
outputs = 10
inputs = 1
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
#
# Test
#
start_time = time.time()
test_accuracy = test(model)
end_time = time.time()
test_time = end_time - start_time
print("="*20)
print("The acc is {}".format(test_accuracy))
print("="*20)
#
# Get signal to noise distribution for all conv layers
#
stn_1 = get_sorted_stn_idx(model.conv1.qw_mean, model.conv1.qw_logvar)
stn_2 = get_sorted_stn_idx(model.conv2.qw_mean, model.conv2.qw_logvar)
if net_type == 'full':
    stn_3 = get_sorted_stn_idx(model.conv3.qw_mean, model.conv3.qw_logvar)
#
# Compute mean and std
#
stn1_mean = np.mean(stn_1[:, 1])
stn1_std = np.std(stn_1[:, 1])

stn2_mean = np.mean(stn_2[:, 1])
stn2_std = np.std(stn_2[:, 1])

if net_type == 'full':
    stn3_mean = np.mean(stn_3[:, 1])
    stn3_std = np.std(stn_2[:, 1])

#
# Save histogram to disk
# (Only for fold 1 and 2 ... to save disk space)
# (only need a few plots)
#
if net_type == 'full' and fold in (0, 1):
    fname = "histogram_{}_{}_{}".format(net_type, config_integer, fold)
    save_hist(stn_1[:, 1], os.path.join(output_dir, fname + "1.png"), 1)
    save_hist(stn_2[:, 1], os.path.join(output_dir, fname + "2.png"), 2)
    save_hist(stn_3[:, 1], os.path.join(output_dir, fname + "3.png"), 3)

# -----------------------------------------------
# Output text file for statistical analysis
#
# We are going to write to 'summary_statistics.txt'
# First line looks like:
# networkType, fold, configInt, learningRate, batchSize, testAccuracy, testTime, stn1mean, stn1var, stn2mean, stn2var, stn3mean, stn3var
# -----------------------------------------------
out_list = list(map(str, [fold,
                          config_integer,
                          learning_rate,
                          batch_size,
                          test_accuracy.item(),
                          test_time,
                          stn1_mean,
                          stn1_std,
                          stn2_mean,
                          stn2_std
                          ]))
if net_type == 'full':
    out_list.append(str(stn3_mean))
    out_list.append(str(stn3_std))

out_string = ", ".join(out_list)

data_file_name = os.path.join(BASE_DIR, "output", "summary_statistics.txt")
data_file = open(data_file_name, 'a')
data_file.write(out_string)
data_file.write("\n")
data_file.close()
