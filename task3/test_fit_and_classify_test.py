#!/usr/bin/env python3

from sys import argv, exit
import numpy as np
from numpy import zeros
from fit_and_classify import fit_and_classify, extract_hog
from skimage.io import imread

import random

import warnings
warnings.filterwarnings("error")


def read_gt(gt_dir):
    fgt = open(gt_dir + '/gt.csv')
    next(fgt)
    lines = fgt.readlines()

    filenames = []
    labels = zeros(len(lines))
    for i, line in enumerate(lines):
        filename, label = line.rstrip('\n').split(',')
        filenames.append(filename)
        labels[i] = int(label)

    return filenames, labels


def extract_features(path, filenames):
    hog_length = len(extract_hog(imread(path + '/' + filenames[0],
                                        plugin='matplotlib')))
    data = zeros((len(filenames), hog_length))
    for i in range(0, len(filenames)):
        filename = path + '/' + filenames[i]
        data[i, :] = extract_hog(imread(filename, plugin='matplotlib'))
    return data

if len(argv) != 3:
    print('Usage: %s train_data_path test_data_path' % argv[0])
    exit(0)

train_data_path = argv[1]
test_data_path = argv[2]

filenames, labels = read_gt(train_data_path)

z = list(zip(filenames, labels))
learn = []
test = []

unique_labels = set(labels)
for l in unique_labels:
    filtered_z = list(filter(lambda t: t[1] == l, z))
    learn += filtered_z[:int(0.8*len(filtered_z))]
    test += filtered_z[int(0.8*len(filtered_z)):]

tlearn = learn.copy()
random.shuffle(tlearn)
tlearn = tlearn[:int(0.2*len(tlearn))]

random.shuffle(learn)
learn = learn[:int(1 * len(learn)) - 1] # limit learning selection

train_filenames = list(list(map(lambda l: l[0], learn)))
train_labels = np.array(list(map(lambda l: l[1], learn)), dtype=np.uint8)

base_train_filenames = list(list(map(lambda l: l[0], tlearn)))
base_train_labels = np.array(list(map(lambda l: l[1], tlearn)), dtype=np.uint8)

test_filenames = list(list(map(lambda l: l[0], test)))
test_labels = np.array(list(map(lambda l: l[1], test)), dtype=np.uint8)

train_features = extract_features(train_data_path, train_filenames)
#base_train_features = extract_features(train_data_path, base_train_filenames)
test_features = extract_features(test_data_path, test_filenames)

y = fit_and_classify(train_features, train_labels, test_features)
# detailed data
diff = np.array(test_labels) - np.array(y)
l = zip(test_labels, y, diff)
l = list(filter(lambda z: z[2] != 0, l))
print(l)
print('Accuracy: %.4f' % (sum(test_labels == y) / float(test_labels.shape[0])))

#y = fit_and_classify(train_features, train_labels, base_train_features)
#print('Accuracy: %.4f'
#      % (sum(base_train_labels == y) / float(base_train_labels.shape[0])))

