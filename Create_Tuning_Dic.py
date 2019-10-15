"""
Create_Tuning_Dic

Simple file creates a dictionary on disk which will originally store
(1) an integer key
(2) hyperparms (learning rate and batch size)
Later:
(3) Accuracy at each fold
(4) Distribution of Signal to Noise Ratio

This is a quick and simple approach, could have also used JSON
"""

import os
import pickle
BASE_DIR = "/homes/murph213/DeepLearning/code_final/"

# Hyperparameter settings and relevant output
hyper_and_out = dict()
key_idx = 1
for lr_ in [0.0001, 0.001, 0.01]:
    for batch_size_ in [64, 128, 256]:
        hyper_and_out[key_idx] = {'lr': lr_, 'batch_size': batch_size_}
        key_idx += 1

fname = os.path.join(BASE_DIR, 'output', 'hyper_and_out.pkl')
print(fname)
pickle.dump(hyper_and_out, open(fname, 'wb'))