import sys
from collections import OrderedDict
import shutil, os
from shutil import copyfile
# from pytorchtools import EarlyStopping
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import pdb
import gc

from sklearn import metrics
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="/home/wangyawei/SIMHEI.TTF", size=14)

import matplotlib.pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
import random
import gzip
import pickle
import timeit
from seq_motifs import get_motif
import argparse

# from sklearn.externals import joblib
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
import argparse

from deap.algorithms import varOr
from torch import optim
import random
import torchvision.transforms as transforms
from torch.autograd import Variable
from deap import base
from deap import creator
from deap import tools

fitss = []
learning_rate = 0.01
mbest = 10
cxpb = 0.5
mutpb = 0.5
mu = 0.5
u = 50
lam = 60
ks = 1
kv = 1
ak = 0.9
bk = 1.1
momentum = 0.9
sigma = []
random.seed(42)
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class CNN(nn.Module):
    def __init__(self, nb_filter, channel=7, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=12, hidden_size=200, stride=(1, 1), padding=0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride=stride)
        out1_size = (window_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1

        maxpool_size = (out1_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1
        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size=(1, 10), stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=stride))
        out2_size = (maxpool_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1
        maxpool2_size = (out2_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1
        self.drop1 = nn.Dropout(p=0.25)
        print('maxpool_size', maxpool_size)
        self.fc1 = nn.Linear(int(maxpool2_size * nb_filter), hidden_size)
        # self.fc1 = nn.Linear(7760, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        torch.cuda.empty_cache()
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        x = x.cpu()
        torch.cuda.empty_cache()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        x = x.cpu()
        torch.cuda.empty_cache()
        return temp[:, 1]


class CNN_LSTM(nn.Module):
    def __init__(self, nb_filter, channel=7, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=12, hidden_size=200, stride=(1, 1), padding=0, num_layers=2):
        super(CNN_LSTM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride=stride)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        out1_size = (window_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1
        maxpool_size = (out1_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1
        self.downsample = nn.Conv2d(nb_filter, 1, kernel_size=(1, 10), stride=stride, padding=padding)
        input_size = (maxpool_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1
        self.layer2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.downsample(out)
        out = torch.squeeze(out, 1)
        # pdb.set_trace()
        if cuda:
            x = x.cuda()
            h0 = Variable(torch.zeros(self.num_layers * 2, out.size(0), self.hidden_size)).cuda()
            c0 = Variable(torch.zeros(self.num_layers * 2, out.size(0), self.hidden_size)).cuda()
        else:
            h0 = Variable(torch.zeros(self.num_layers * 2, out.size(0), self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers * 2, out.size(0), self.hidden_size))
        out, _ = self.layer2(out, (h0, c0))
        out = out[:, -1, :]
        # pdb.set_trace()
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        x = x.cpu()
        h0 = h0.cpu()
        c0 = c0.cpu()
        torch.cuda.empty_cache()
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        x = x.cpu()
        torch.cuda.empty_cache()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        x = x.cpu()
        torch.cuda.empty_cache()
        return temp[:, 1]


net1 = CNN(nb_filter=16, labcounts=4, window_size=507, channel=1)
net2 = CNN(nb_filter=16, labcounts=4, window_size=107, channel=7)

EPOCH = 160
pre_epoch = 0
BATCH_SIZE = 125
LR = 0.1


def GENnetwork():
    a = [net1.state_dict(), net2.state_dict()]
    return a


def mate1(toolbox, ind1, ind2):
    for k, v in ind1.items():
        if str(k).find('num_batches_tracked', 0, len(k)) == -1:
            ind1[k], ind2[k] = toolbox.mate(ind1[k], ind2[k])
    return ind1, ind2


def mate2(toolbox, ind1, ind2):
    mate1(toolbox, ind1[0], ind2[0])
    mate1(toolbox, ind1[1], ind2[1])
    return ind1, ind2


def mutate1(ind, generation):
    sigma1 = 0.01 / (generation + 1)
    # sigma1 = 0.01-0.0005*generation
    for k, v in ind.items():
        if str(k).find('num_batches_tracked', 0, len(k)) == -1:
            ind[k] = tools.mutGaussian(ind[k], 0, sigma1, indpb=0.05)
    return ind


def mutate2(ind, generation):
    ind[0] = mutate1(ind[0], generation)
    ind[1] = mutate1(ind[1], generation)
    return ind


def recombin1(ind, ind1, ind2):
    for k, v in ind1.items():
        if str(k).find('num_batches_tracked', 0, len(k)) == -1:
            ind[k] = (ind1[k] + ind2[k]) / 2
    return ind


def recombin2(ind, ind1, ind2):
    ind[0] = recombin1(ind[0], ind1[0], ind2[0])
    ind[1] = recombin1(ind[1], ind1[1], ind2[1])
    return ind


def recombin_mutate(toolbox, ind, ind1, ind2):
    ind = recombin2(ind, ind1, ind2)
    ind = mutate2(ind, generation)
    return ind


def evalFitness(ind, X, y, batch_size=32):
    y_pred = ind.predict(X)

    y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)
    if cuda:
        y_v = y_v.cuda()
    loss = ind.loss_f(y_pred, y_v)
    predict = y_pred.data.cpu().numpy()[:, 1].flatten()
    auc = roc_auc_score(y, predict)
    # print(auc)
    # lasses = torch.topk(y_pred, 1)[1].data.numpy().flatten()
    # cc = self._accuracy(classes, y)
    # loss.item(), auc
    return auc,


def generate_offspring(population, toolbox, lambda_, cxpb, mutpb, generation):
    offspring = []
    p = []
    for i in range(50):
        p.append(0.0004 + 0.0008 * (49 - i))
    p = np.array(p)
    index_p = []
    for i in range(50):
        index_p.append(i)
    for _ in range(lambda_):
        index1 = np.random.choice(index_p, p=p.ravel())
        index2 = np.random.choice(index_p, p=p.ravel())
        print(index1)
        print(index2)
        aa = []
        aa.append(population[index1])
        aa.append(population[index2])
        ind3, ind4 = list(map(toolbox.clone, random.sample(aa, 2)))
        ind = toolbox.clone(random.choice(population))
        ind = recombin_mutate(toolbox, ind, ind3, ind4)
        # del ind.fitness.values
        offspring.append(ind)

    return offspring


dim = 100
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("network", GENnetwork)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.network)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, 0, 0.01, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=1)
# 可以设置多个目标，

toolbox.register("evaluate", evalFitness)


def cpy(x, y):
    for k, v in x.items():
        x[k] = y[k]


def fitscore(individual):
    return individual.fitness.values


criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    cuda = True
    # torch.cuda.set_device(1)
    print('===> Using GPU')
else:
    cuda = False
    print('===> Using CPU')


# cuda = False
def padding_sequence_new(seq, max_len=101, repkey='N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < max_len:
        gap_len = max_len - seq_len
        new_seq = seq + repkey * gap_len
    return new_seq


def read_rna_dict(rna_dict='rna_dict'):
    odr_dict = {}
    with open(rna_dict, 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',')
            for ind, val in enumerate(values):
                val = val.strip()
                odr_dict[val] = ind

    return odr_dict


def padding_sequence(seq, max_len=501, repkey='N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len - seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq


def get_RNA_seq_concolutional_array(seq, motif_len=4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    # for seq in seqs:
    # for key, seq in seqs.iteritems():
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - 3, row):
        new_array[i] = np.array([0.25] * 4)

    # pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        # if val == 'N' or i < motif_len or i > len(seq) - motif_len:
        #    new_array[i] = np.array([0.25]*4)
        # else:
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        # data[key] = new_array
    return new_array


def split_overlap_seq(seq, window_size=101):
    overlap_size = 20
    # pdb.set_trace()
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size) / (window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size) % (window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            # pdb.set_trace()
            # start = len(seq) -window_size
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, max_len=window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs


def read_seq_graphprot(seq_file, label=1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)

    return seq_list, labels


def get_RNA_concolutional_array(seq, motif_len=4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    # for seq in seqs:
    # for key, seq in seqs.iteritems():
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - 3, row):
        new_array[i] = np.array([0.25] * 4)

    # pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        # data[key] = new_array
    return new_array


def load_graphprot_data(protein, train=True, path='./GraphProt_CLIP_sequences/'):
    data = dict()
    tmp = []
    listfiles = os.listdir(path)

    key = '.train.'
    if not train:
        key = '.ls.'
    mix_label = []
    mix_seq = []
    mix_structure = []
    for tmpfile in listfiles:
        if protein not in tmpfile:
            continue
        if key in tmpfile:
            if 'positive' in tmpfile:
                label = 1
            else:
                label = 0
            _path = os.path.join(path, tmpfile)
            seqs, labels = read_seq_graphprot(_path, label=label)
            # pdb.set_trace()
            mix_label = mix_label + labels
            mix_seq = mix_seq + seqs

    data["seq"] = mix_seq
    data["Y"] = np.array(mix_label)

    return data


def loaddata_graphprot(protein, train=True, ushuffle=True):
    # pdb.set_trace()
    data = load_graphprot_data(protein, train=train)
    label = data["Y"]
    rna_array = []
    # trids = get_6_trids()
    # nn_dict = read_rna_dict()
    for rna_seq in data["seq"]:
        # rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')

        seq_array = get_RNA_seq_concolutional_array(seq)
        # tri_feature = get_6_nucleotide_composition(trids, rna_seq_pad, nn_dict)
        rna_array.append(seq_array)

    return np.array(rna_array), label


def get_bag_data(data, channel=7, window_size=101):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        # pdb.set_trace()
        bag_seqs = split_overlap_seq(seq, window_size=window_size)
        # flat_array = []
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)

        if num_of_ins > channel:
            start = (num_of_ins - channel) / 2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) < channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                # bag_subt.append(random.choice(bag_subt))
                tri_fea = get_RNA_seq_concolutional_array('N' * window_size)
                bag_subt.append(tri_fea.T)

        bags.append(np.array(bag_subt))

    return bags, labels
    # for data in pairs.iteritems():
    #    ind1 = trids.index(key)
    #    emd_weight1 = embedding_rna_weights[ord_dict[str(ind1)]]


def get_bag_data_1_channel(data, max_len=501):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        # pdb.set_trace()
        # bag_seqs = split_overlap_seq(seq)
        bag_seq = padding_sequence(seq, max_len=max_len)
        # flat_array = []
        bag_subt = []
        # for bag_seq in bag_seqs:
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        bag_subt.append(tri_fea.T)

        bags.append(np.array(bag_subt))

    return bags, labels


def batch(tensor, batch_size=1000):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i + 1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i + 1) * batch_size])
        i += 1


class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, train_loader):
        """
        train one epoch
        """
        loss_list = []
        acc_list = []
        for idx, (X, y) in enumerate(train_loader):
            X_v = Variable(X)
            y_v = Variable(y)
            if cuda:
                X_v = X_v.cuda()
                y_v = y_v.cuda()

            self.optimizer.zero_grad()
            y_pred = self.model(X_v)
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            ## for log
            loss_list.append(loss.item())  # need change to loss_list.append(loss.item()) for pytorch v0.4 or above
            if cuda:
                X_v = X_v.cpu()
                y_v = y_v.cpu()
                torch.cuda.empty_cache()

        return sum(loss_list) / len(loss_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        # X_list = batch(X, batch_size)
        # y_list = batch(y, batch_size)
        # pdb.set_trace()
        print(X.shape)
        train_set = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                                  torch.from_numpy(y.astype(np.float32)).long().view(-1))
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        self.model.train()
        for t in range(nb_epoch):
            loss = self._fit(train_loader)
            print(loss)

            # self.model.eval()
            #
            # rint("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))

    def evaluate(ind):
        auc = 0
        return auc

    def _accuracy(self, y_pred, y):
        return float(sum(y_pred == y)) / y.shape[0]

    def predict(self, X):
        X = Variable(torch.from_numpy(X.astype(np.float32)))
        if cuda:
            X = X.cuda()
        y_pred = self.model(X)
        X = X.cpu()
        torch.cuda.empty_cache()
        return y_pred

    def predict_proba(X):
        self.model.eval()
        return self.model.predict_proba(X)


def convR(in_channels, out_channels, kernel_size, stride=1, padding=(0, 1)):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     padding=padding, stride=stride, bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, nb_filter=16, kernel_size=(1, 3), stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = convR(in_channel, nb_filter, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convR(nb_filter, nb_filter, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(nb_filter)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, nb_filter=16, channel=7, labcounts=12, window_size=36, kernel_size=(1, 3),
                 pool_size=(1, 3), num_classes=2, hidden_size=200):
        super(ResNet, self).__init__()
        self.in_channels = channel
        self.conv = convR(self.in_channels, nb_filter, kernel_size=(4, 10))
        cnn1_size = window_size - 7
        self.bn = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, nb_filter, layers[0], kernel_size=kernel_size)
        self.layer2 = self.make_layer(block, nb_filter * 2, layers[1], 1, kernel_size=kernel_size,
                                      in_channels=nb_filter)
        self.layer3 = self.make_layer(block, nb_filter * 4, layers[2], 1, kernel_size=kernel_size,
                                      in_channels=2 * nb_filter)
        self.avg_pool = nn.AvgPool2d(pool_size)
        avgpool2_1_size = (cnn1_size - (pool_size[1] - 1) - 1) / pool_size[1] + 1
        last_layer_size = 4 * nb_filter * avgpool2_1_size
        self.fc = nn.Linear(last_layer_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1, kernel_size=(1, 10), in_channels=16):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                convR(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, downsample=downsample))
        # self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print x.data.cpu().numpy().shape
        # x = x.view(x.size(0), 1, x.size(1), x.size(2))
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        # pdb.set_trace()
        # print self.layer2
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        # pdb.set_trace()
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        torch.cuda.empty_cache()
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        x = x.cpu()
        torch.cuda.empty_cache()
        temp = out.data.cpu().numpy()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        x = x.cpu()
        torch.cuda.empty_cache()
        temp = y.data.cpu().numpy()
        return temp[:, 1]


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                                            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseNet(nn.Module):
    def __init__(self, labcounts=4, window_size=107, channel=7, growth_rate=6, block_config=(16, 16, 16),
                 compression=0.5,
                 num_init_features=12, bn_size=2, drop_rate=0, avgpool_size=(1, 8),
                 num_classes=2):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = avgpool_size

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(channel, num_init_features, kernel_size=(4, 10), stride=1, bias=False)),
        ]))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        last_size = window_size - 7
        # Each denseblock
        num_features = num_init_features
        # last_size =  window_size
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features
                                                            * compression))
                last_size = (last_size - (2 - 1) - 1) / 2 + 1
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        avgpool2_1_size = (last_size - (self.avgpool_size[1] - 1) - 1) / self.avgpool_size[1] + 1
        num_features = num_features * avgpool2_1_size
        print(num_features)
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # x = x.view(x.size(0), 1, x.size(1), x.size(2))
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
            features.size(0), -1)
        out = self.classifier(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.features[0](x)
        out = self.features[1](out)
        temp = out.data.cpu().numpy()
        x = x.cpu()
        torch.cuda.empty_cache()
        return temp

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        x = x.cpu()
        torch.cuda.empty_cache()
        return temp[:, 1]


def get_all_data(protein, channel=7):
    data = load_graphprot_data(protein)
    test_data = load_graphprot_data(protein, train=False)
    # pdb.set_trace()
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data)
        test_bags, true_y = get_bag_data_1_channel(test_data)
    else:
        train_bags, label = get_bag_data(data)
        # pdb.set_trace()
        test_bags, true_y = get_bag_data(test_data)

    return train_bags, label, test_bags, true_y


def run_network(model_type, X_train, test_bags, y_train, channel=7, window_size=107):
    print('model training for ', model_type)
    print("run_network")
    # nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter=16, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter=16, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter=16, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'DenseNet':
        model = DenseNet(window_size=window_size, channel=channel, labcounts=4)
    else:
        print('only support CNN, CNN-LSTM, ResNet and DenseNet model')

    if cuda:
        model = model.cuda()
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001),
                loss=nn.CrossEntropyLoss())
    clf.fit(X_train, y_train, batch_size=100, nb_epoch=50)

    print('predicting')
    pred = model.predict_proba(test_bags)
    model = model.cpu()
    torch.cuda.empty_cache()
    return pred, model


def run_ideepe_on_graphprot(model_type='CNN', local=False, ensemble=True):
    data_dir = './GraphProt_CLIP_sequences/'

    finished_protein = set()
    start_time = timeit.default_timer()
    if local:
        window_size = 107
        channel = 7
        lotext = 'local'
    else:
        window_size = 507
        channel = 1
        lotext = 'global'
    if ensemble:
        outputfile = lotext + '_result_adam_ensemble_' + model_type
    else:
        outputfile = lotext + '_result_adam_individual_' + model_type

    fw = open(outputfile, 'w')

    for protein in os.listdir(data_dir):
        protein = protein.split('.')[0]
        if protein in finished_protein:
            continue
        finished_protein.add(protein)
        print(protein)
        fw.write(protein + '\t')
        hid = 16
        if not ensemble:
            train_bags, train_labels, test_bags, test_labels = get_all_data(protein, channel=channel)
            predict = run_network(model_type, np.array(train_bags), np.array(test_bags), np.array(train_labels),
                                  protein, channel=channel, window_size=window_size)
        else:
            print('ensembling')
            train_bags, train_labels, test_bags, test_labels = get_all_data(protein, channel=1)
            predict1 = run_network(model_type, np.array(train_bags), np.array(test_bags), np.array(train_labels),
                                   channel=1, window_size=507)
            train_bags, train_labels, test_bags, test_labels = [], [], [], []
            train_bags, train_labels, test_bags, test_labels = get_all_data(protein, channel=7)
            predict2 = run_network(model_type, np.array(train_bags), np.array(test_bags), np.array(train_labels),
                                   channel=7, window_size=107)
            predict = (predict1 + predict2) / 2.0
            train_bags, train_labels, test_bags = [], [], []

        auc = roc_auc_score(test_labels, predict)
        print('AUC:', auc)
        fw.write(str(auc) + '\n')
        mylabel = "\t".join(map(str, test_labels))
        myprob = "\t".join(map(str, predict))
        fw.write(mylabel + '\n')
        fw.write(myprob + '\n')
    fw.close()
    end_time = timeit.default_timer()
    print("Training final took: %.2f s" % (end_time - start_time))


def read_data_file(posifile, negafile=None, train=True):
    data = dict()
    seqs, labels = read_seq_graphprot(posifile, label=1)
    if negafile:
        seqs2, labels2 = read_seq_graphprot(negafile, label=0)
        seqs = seqs + seqs2
        labels = labels + labels2

    data["seq"] = seqs
    data["Y"] = np.array(labels)

    return data


def get_data(posi, nega=None, channel=7, window_size=101, train=True):
    data = read_data_file(posi, nega, train=train)
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data, max_len=window_size)

    else:
        train_bags, label = get_bag_data(data, channel=channel, window_size=window_size)

    return train_bags, label


def detect_motifs(model, test_seqs, X_train, output_dir='motifs', channel=1):
    if channel == 1:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for param in model.parameters():
            layer1_para = param.data.cpu().numpy()
            break
        # test_data = load_graphprot_data(protein, train = True)
        # test_seqs = test_data["seq"]
        N = len(test_seqs)
        if N > 15000:  # do need all sequence to generate motifs and avoid out-of-memory
            sele = 15000
        else:
            sele = N
        ix_all = np.arange(N)
        np.random.shuffle(ix_all)
        ix_test = ix_all[0:sele]

        X_train = X_train[ix_test, :, :, :]
        test_seq = []
        for ind in ix_test:
            test_seq.append(test_seqs[ind])
        test_seqs = test_seq
        filter_outs = model.layer1out(X_train)[:, :, 0, :]
        get_motif(layer1_para[:, 0, :, :], filter_outs, test_seqs, dir1=output_dir)


def train_network(model_type, X_train, y_train, channel=7, window_size=107, model_file='model.pkl', batch_size=100,
                  n_epochs=50, num_filters=16, motif=False, motif_seqs=[], motif_outdir='motifs'):
    print('model training for ', model_type)
    # nb_epos= 5

    if model_type == 'CNN':
        model = CNN(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter=num_filters, labcounts=4, channel=channel,
                       window_size=window_size)
    elif model_type == 'DenseNet':
        model = DenseNet(window_size=window_size, channel=channel, labcounts=4)
    else:
        print('only support CNN, CNN-LSTM, ResNet and DenseNet model')
    # model.fc1 = nn.Linear(int(1360), 200)
    # model.fc2 = nn.Linear(200, 2)
    cwd = os.getcwd()
    path_ = cwd + '/' + model_file
    if os.path.isfile(path_):
        new_model = torch.load(model_file)
        model.load_state_dict(new_model)

    if cuda:
        model = model.cuda()
    model.eval()
    if motif and channel == 1:
        detect_motifs(model, motif_seqs, X_train, motif_outdir)
    else:
        clf = Estimator(model)
        # learning_rate1 = random.uniform(0.00009, 0.00011)
        rand = random.random()
        lr1 = random.uniform(0.1, 1)
        lr2 = random.uniform(1e-3, 1e-2)
        lr3 = random.uniform(1e-4, 1e-3)
        mt = random.uniform(0.9, 0.99)
        # lr1 = 1
        # lr2 = 1e-3
        # lr3 = 1e-4
        wd = random.uniform(1e-5, 1e-4)
        if rand < 0.2:
            clf.compile(optimizer=torch.optim.SGD(model.parameters(), lr=lr2, momentum=mt, weight_decay=wd),
                        loss=nn.CrossEntropyLoss())
        elif rand < 0.3:
            clf.compile(optimizer=torch.optim.Adagrad(model.parameters(), lr=lr2, weight_decay=wd),
                        loss=nn.CrossEntropyLoss())  # 1e-2,lr3 work
        elif rand < 0.5:
            clf.compile(optimizer=torch.optim.RMSprop(model.parameters(), lr=lr3, momentum=mt, weight_decay=wd),
                        loss=nn.CrossEntropyLoss())  # 1e-2,1e-3 not juge,1e-4 work
        elif rand < 0.7:
            clf.compile(optimizer=torch.optim.Adadelta(model.parameters(), lr=lr1, weight_decay=wd),
                        loss=nn.CrossEntropyLoss())  # la ji youhuaqi
        elif rand < 0.8:
            clf.compile(optimizer=torch.optim.Adamax(model.parameters(), lr=lr3, weight_decay=wd),
                        loss=nn.CrossEntropyLoss())  # 2e-3, not juge,1e-4 work
        else:
            clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=lr3, weight_decay=wd),
                        loss=nn.CrossEntropyLoss())  # 1e-3,1e-4 work
        clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5)
        torch.save(model.state_dict(), "tmp" + model_file)
        model = model.cpu()
        torch.cuda.empty_cache()
        return model
    # print('predicting')
    # pred = model.predict_proba(test_bags)
    # return model,auc1


def predict_network(model_type, X_test, channel=7, window_size=107, model_file='model.pkl', batch_size=100, n_epochs=50,
                    num_filters=16):
    print('model training for ', model_type)
    # nb_epos= 5
    print(window_size)
    print(num_filters)
    print(channel)
    print(model_file)
    if model_type == 'CNN':
        model = CNN(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter=num_filters, labcounts=4, channel=channel,
                       window_size=window_size)
    elif model_type == 'DenseNet':
        model = DenseNet(window_size=window_size, channel=channel, labcounts=4)
    else:
        print('only support CNN, CNN-LSTM, ResNet and DenseNet model')

    if cuda:
        model = model.cuda()
    print(model.fc1)
    model.load_state_dict(torch.load(model_file))
    # cpy(model.state_dict(),torch.load(model_file))
    model.eval()
    try:
        pred = model.predict_proba(X_test)
        torch.cuda.empty_cache()
    except:  # to handle the out-of-memory when testing
        test_batch = batch(X_test)
        pred = []
        for test in test_batch:
            pred_test1 = model.predict_proba(test)[:, 1]
            pred = np.concatenate((pred, pred_test1), axis=0)
            torch.cuda.empty_cache()
    model = model.cpu()
    torch.cuda.empty_cache()
    del model
    gc.collect()
    return pred


def individual_evaluation(model_type, X_test, channel=7, window_size=107, individual=None, batch_size=100, n_epochs=50,
                      num_filters=16):
    print('model training for ', model_type)
    # nb_epos= 5
    print(window_size)
    print(num_filters)
    print(channel)
    if model_type == 'CNN':
        model = CNN(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter=num_filters, labcounts=4, channel=channel,
                       window_size=window_size)
    elif model_type == 'DenseNet':
        model = DenseNet(window_size=window_size, channel=channel, labcounts=4)
    else:
        print('only support CNN, CNN-LSTM, ResNet and DenseNet model')

    if cuda:
        model = model.cuda()
    s = (12, 34.56)
    for k, v in individual.items():
        if type(individual[k]) == type(s):
            individual[k] = individual[k][0]
    model.load_state_dict(individual)
    model.eval()
    try:
        pred = model.predict_proba(X_test)
        torch.cuda.empty_cache()
    except:  # to handle the out-of-memory when testing
        test_batch = batch(X_test)
        pred = []
        for test in test_batch:
            pred_test1 = model.predict_proba(test)[:, 1]
            pred = np.concatenate((pred, pred_test1), axis=0)
            torch.cuda.empty_cache()
    model = model.cpu()
    torch.cuda.empty_cache()
    del model
    gc.collect()
    return pred


def readname1():
    # filePath = 'C:\\Users\\wangyawei\\Desktop\\iDeepE-master\\GraphProt_CLIP_sequences\\'
    # cwd = os.getcwd()
    name = os.listdir('GraphProt_CLIP_sequences')
    return name


def readname():
    cwd = os.getcwd()
    filePath = cwd + '/GraphProt_CLIP_sequences'
    name = os.listdir(filePath)
    return name


def run_edcnn(parser):
    data_dir = './GraphProt_CLIP_sequences/'
    global generation
    population_size = parser.population_size
    generation_number = parser.generation_number
    posi = parser.posi
    nega = parser.nega
    model_type = parser.model_type
    ensemble = parser.ensemble
    out_file = parser.out_file
    train = parser.train
    model_file = parser.model_file
    predict = parser.predict
    motif = parser.motif
    motif_outdir = parser.motif_dir
    max_size = parser.maxsize
    channel = parser.channel
    local = parser.local
    window_size = parser.window_size
    ensemble = parser.ensemble
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    num_filters = parser.num_filters
    testfile = parser.testfile
    glob = parser.glob
    start_time = timeit.default_timer()
    # pdb.set_trace()
    if predict:
        train = False
        if posi == '' or nega == '':
            print('you need specify the fasta file for predicting when predict is True')
            return
    if train:
        if posi == '' or nega == '':
            print('you need specify the training positive and negative fasta file for training when train is True')
            return
    motif_seqs = []
    if motif:
        train = True
        local = False
        glob = True
        # pdb.set_trace()
        data = read_data_file(posi, nega)
        motif_seqs = data['seq']
        if posi == '' or nega == '':
            print('To identify motifs, you need training positive and negative sequences using global CNNs.')
            return

    if local:
        # window_size = window_size + 6
        channel = channel
        ensemble = False
        # lotext = 'local'
    elif glob:
        # window_size = maxsize + 6
        channel = 1
        ensemble = False
        window_size = max_size
        # lotext = 'global'
    # if local and ensemble:
    #	ensemble = False

    if train:
        if not ensemble:
            train_bags, train_labels = get_data(posi, nega, channel=channel, window_size=window_size)
            model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel=channel,
                                  window_size=window_size + 6,
                                  model_file=model_file, batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, motif=motif, motif_seqs=motif_seqs,
                                  motif_outdir=motif_outdir)
        else:
            print('ensembling')
            t1 = posi.find('/')
            t2 = posi.find('.')
            file_name = posi[t1 + 1:t2]
            best_score = 0.5
            sgd = file_name + ".sgd.txt"
            mutate = file_name + ".mutate.txt"
            best_acc = file_name + ".best_acc.txt"
            model1 = CNN(nb_filter=num_filters, labcounts=4, window_size=window_size + 6,
                         channel=1)
            model2 = model = CNN(nb_filter=num_filters, labcounts=4, window_size=window_size + 6,
                                 channel=7)
            torch.save(model1.state_dict(), file_name + '.' + 'best' + '.' + model_file + '.global')
            torch.save(model2.state_dict(), file_name + '.' + 'best' + '.' + model_file + '.local')
            del model2
            del model1
            gc.collect()
            with open(sgd, "w") as f1:
                with open(mutate, "w") as f2:
                    with open(best_acc, "w") as f3:
                        population = toolbox.population(u)

                        # global population
                        posi1 = 'GraphProt_CLIP_sequences/' + file_name + '.ls.positives.fa'
                        nega1 = 'GraphProt_CLIP_sequences/' + file_name + '.ls.negatives.fa'
                        data = read_data_file(posi, nega, train=train)
                        rna_seqs = data["seq"]
                        labels = data["Y"]
                        train_seqs, val_seqs, train_label, val_label = train_test_split(rna_seqs, labels,random_state=42,
                                                                                          test_size=0.1)
                        data1 = dict()
                        data1["seq"] = train_seqs
                        data1["Y"] = train_label
                        train_bags1, train_labels1 = get_bag_data_1_channel(data1, max_len=max_size)
                        train_bags2, train_labels2 = get_bag_data(data1, channel=7, window_size=window_size)

                        data2 = dict()
                        data2["seq"] = val_seqs
                        data2["Y"] = val_label
                        X_test3,X_labels3 = get_bag_data_1_channel(data2, max_len=max_size)
                        X_test4, X_labels4 = get_bag_data(data2, channel=7, window_size=window_size)
                        for k in range(generation_number):
                            generation = k
                            print('generation %d' % (k))
                            for j in range(population_size):
                                print('individual:%d' % (j))
                                model_1 = CNN(nb_filter=num_filters, labcounts=4, window_size=window_size + 6,
                                              channel=1)
                                model_1 = train_network(model_type, np.array(train_bags1), np.array(train_labels1),
                                                        channel=1,
                                                        window_size=max_size + 6,
                                                        model_file=file_name + '.' + str(
                                                            j) + '.' + model_file + '.global',
                                                        batch_size=batch_size,
                                                        n_epochs=n_epochs,
                                                        num_filters=num_filters, motif=motif, motif_seqs=motif_seqs,
                                                        motif_outdir=motif_outdir)

                                model_2 = CNN(nb_filter=num_filters, labcounts=4, window_size=window_size + 6,
                                              channel=7)
                                model_2 = train_network(model_type, np.array(train_bags2), np.array(train_labels2),
                                                        channel=7,
                                                        window_size=window_size + 6,
                                                        model_file=file_name + '.' + str(
                                                            j) + '.' + model_file + '.local',
                                                        batch_size=batch_size,
                                                        n_epochs=n_epochs,
                                                        num_filters=num_filters, motif=motif, motif_seqs=motif_seqs,
                                                        motif_outdir=motif_outdir)

                                predict1 = predict_network(model_type, np.array(X_test3), channel=1,
                                                           window_size=max_size + 6,
                                                           model_file='tmp' + file_name + '.' + str(
                                                               j) + '.' + model_file + '.global',
                                                           batch_size=batch_size,
                                                           n_epochs=n_epochs,
                                                           num_filters=num_filters)
                                predict2 = predict_network(model_type, np.array(X_test4), channel=7,
                                                           window_size=window_size + 6,
                                                           model_file='tmp' + file_name + '.' + str(
                                                               j) + '.' + model_file + '.local',
                                                           batch_size=batch_size,
                                                           n_epochs=n_epochs,
                                                           num_filters=num_filters)
                                predict = (predict1 + predict2) / 2.0
                                auc = roc_auc_score(X_labels4, predict)
                                gg = []
                                gg.append(auc)
                                print(auc)
                                print('population.fitness.values', population[j].fitness.values)
                                os.remove('tmp' + file_name + '.' + str(
                                    j) + '.' + model_file + '.global')
                                os.remove('tmp' + file_name + '.' + str(
                                    j) + '.' + model_file + '.local')
                                if k == 0:
                                    # print("1126")
                                    population[j][0] = model_1.state_dict()
                                    population[j][1] = model_2.state_dict()
                                    population[j].fitness.values = gg
                                elif auc > population[j].fitness.values:
                                    # print("1125")
                                    population[j][0] = model_1.state_dict()
                                    population[j][1] = model_2.state_dict()
                                    population[j].fitness.values = gg
                                torch.cuda.empty_cache()
                            for v in range(kv):
                                for ind in population:
                                    predict1 = individual_evaluation(model_type, np.array(X_test3), channel=1,
                                                                 window_size=max_size + 6,
                                                                 individual=ind[0],
                                                                 batch_size=batch_size,
                                                                 n_epochs=n_epochs,
                                                                 num_filters=num_filters)
                                    predict2 = individual_evaluation(model_type, np.array(X_test4), channel=7,
                                                                 window_size=window_size + 6,
                                                                 individual=ind[1],
                                                                 batch_size=batch_size,
                                                                 n_epochs=n_epochs,
                                                                 num_filters=num_filters)
                                    predict = (predict1 + predict2) / 2.0
                                    auc = roc_auc_score(X_labels4, predict)
                                    gg = []
                                    gg.append(auc)
                                    ind.fitness.values = gg
                                    torch.cuda.empty_cache()
                                population.sort(key=fitscore, reverse=True)
                                offspring = generate_offspring(population, toolbox, 4 * u, cxpb, mutpb, k)
                                for ind in offspring:
                                    predict1 = individual_evaluation(model_type, np.array(X_test3), channel=1,
                                                                 window_size=max_size + 6,
                                                                 individual=ind[0],
                                                                 batch_size=batch_size,
                                                                 n_epochs=n_epochs,
                                                                 num_filters=num_filters)
                                    predict2 = individual_evaluation(model_type, np.array(X_test4), channel=7,
                                                                 window_size=window_size + 6,
                                                                 individual=ind[1],
                                                                 batch_size=batch_size,
                                                                 n_epochs=n_epochs,
                                                                 num_filters=num_filters)
                                    predict = (predict1 + predict2) / 2.0
                                    auc = roc_auc_score(X_labels4, predict)
                                    gg = []
                                    gg.append(auc)
                                    ind.fitness.values = gg
                                    torch.cuda.empty_cache()
                                population = population + offspring
                                del offspring
                                gc.collect()
                                print('fitness populaton1')
                                for ind in population:
                                    print(ind.fitness)
                                m = int(0.6 * u)
                                population.sort(key=fitscore, reverse=True)
                                elist = tools.selBest(population, m)
                                population = elist + tools.selRandom(population[m:], u - m)
                                print('fitness population1')
                                for ind in population:
                                    print(ind.fitness)
                                cnt = 0
                                for ind in population:
                                    torch.save(ind[0], file_name + '.' + str(
                                        cnt) + '.' + model_file + '.global')
                                    torch.save(ind[1], file_name + '.' + str(
                                        cnt) + '.' + model_file + '.local')
                                    cnt = cnt + 1
                                torch.cuda.empty_cache()

                        with torch.no_grad():
                            X_test1, X_labels1 = get_data(posi1, nega1, channel=1, window_size=max_size)
                            X_test2, X_labels2 = get_data(posi1, nega1, channel=7, window_size=window_size)
                            auclist = []
                            for j in range(population_size):
                                predict1 = predict_network(model_type, np.array(X_test1), channel=1,
                                                           window_size=max_size + 6,
                                                           model_file=file_name + '.' + str(
                                                               j) + '.' + model_file + '.global',
                                                           batch_size=batch_size,
                                                           n_epochs=n_epochs,
                                                           num_filters=num_filters)
                                predict2 = predict_network(model_type, np.array(X_test2), channel=7,
                                                           window_size=window_size + 6,
                                                           model_file=file_name + '.' + str(
                                                               j) + '.' + model_file + '.local',
                                                           batch_size=batch_size,
                                                           n_epochs=n_epochs,
                                                           num_filters=num_filters)
                                predict = (predict1 + predict2) / 2.0
                                auc = roc_auc_score(X_labels2, predict)
                                auclist.append(auc)
                                if auc > best_score:
                                    best_score = auc
                                    copyfile(file_name + '.' + str(j) + '.' + model_file + '.global',
                                             file_name + '.' + 'best' + '.' + model_file + '.global')
                                    copyfile(file_name + '.' + str(j) + '.' + model_file + '.local',
                                             file_name + '.' + 'best' + '.' + model_file + '.local')
                                    f3.write(str(best_score) + '\n')
                                torch.cuda.empty_cache()
                            auclist.sort(reverse=True)
                            for a in auclist:
                                print("mutate" + str(a))
                                f1.write(str(a) + '\n')
                            del X_test1, X_labels1
                            del X_test2, X_labels2
                            gc.collect()
                            torch.cuda.empty_cache()

                        del train_bags1, train_labels1
                        del train_bags2, train_labels2
                        # del train_bags1_cp, X_test3, train_labels1_cp, X_labels3
                        # del train_bags2_cp, X_test4, train_labels2_cp, X_labels4
                        del X_test3, X_labels3, X_test4, X_labels4
                        gc.collect()
                        torch.cuda.empty_cache()
                        f3.close()
                    f2.close()
                f1.close()
            end_time = timeit.default_timer()
            print("Training final took: %.2f s" % (end_time - start_time))

    elif predict:
        fw = open(out_file, 'w')
        if not ensemble:
            X_test, X_labels = get_data(testfile, nega=None, channel=channel, window_size=window_size)
            predict = predict_network(model_type, np.array(X_test), channel=channel, window_size=window_size + 6,
                                      model_file=model_file, batch_size=batch_size, n_epochs=n_epochs,
                                      num_filters=num_filters)
        else:
            print('haha')
            posis = []
            negas = []
            name = readname1()
            print(name)
            cwd = os.getcwd()
            for tt in name:
                if tt.find('ls') != -1:
                    if tt.find('negatives') != -1:
                        negas.append('GraphProt_CLIP_sequences' + '/' + tt)
                    else:
                        posis.append('GraphProt_CLIP_sequences' + '/' + tt)
                    # print(path)
                    # tot = tot+1
            posis.sort()
            negas.sort()
            sum1 = 0
            filenames = ['ALKBH5(AUC=0.768)',
                         'C17ORF85(AUC=0.88)',
                         'C22ORF28(AUC=0.869)',
                         'CAPRIN1(AUC=0.912)',
                         'Ago2(AUC=0.895)',
                         'ELAVL1H(AUC=0.981)',
                         'SFRS1(AUC=0.957)',
                         'HNRNPC(AUC=0.983)',
                         'TDP43(AUC=0.955)',
                         'TIA1(AUC=0.95)',
                         'TIAL1(AUC=0.944)',
                         'Ago1-4(AUC=0.934)',
                         'ELAVL1B(AUC=0.982)',
                         'ELAVL1A(AUC=0.977)',
                         'EWSR1(AUC=0.976)',
                         'FUS(AUC=0.988)',
                         'ELAVL1C(AUC=0.993)',
                         'IGF2BP1-3(AUC=0.969)',
                         'MOV10(AUC=0.94)',
                         'PUM2(AUC=0.974)',
                         'QKI(AUC=0.973)',
                         'TAF15(AUC=0.982)',
                         'PTB(AUC=0.954)',
                         'ZC3H7B(AUC=0.92)'
                         ]
            auc233 = []
            for gog in range(len(posis)):
                posi = posis[gog]
                nega = negas[gog]
                t1 = posi.find('/')
                t2 = posi.find('.')
                file_name = posi[t1 + 1:t2]
                X_test, X_labels = get_data(posi, nega, channel=1, window_size=max_size)
                predict1 = predict_network(model_type, np.array(X_test), channel=1, window_size=max_size + 6,
                                           model_file=file_name + '.' + 'best' + '.' + model_file + '.global',
                                           batch_size=batch_size,
                                           n_epochs=n_epochs,
                                           num_filters=num_filters)
                X_test, X_labels = get_data(posi, nega, channel=7, window_size=window_size)
                predict2 = predict_network(model_type, np.array(X_test), channel=7, window_size=window_size + 6,
                                           model_file=file_name + '.' + 'best' + '.' + model_file + '.local',
                                           batch_size=batch_size,
                                           n_epochs=n_epochs,
                                           num_filters=num_filters)
                predict = (predict1 + predict2) / 2.0
                # print(X_labels)
                ff = open("roc.txt", "w")
                ff.write(str(predict))
                ff.write(str(X_labels))
                auc = roc_auc_score(X_labels, predict)
                auc233.append(auc)
                sum1 = sum1 + auc
                # if gog<12:
                fpr, tpr, thresholds = metrics.roc_curve(X_labels, predict, pos_label=1)
                # else:
                #   fpr, tpr, thresholds = metrics.roc_curve(X_labels, predict, pos_label=2)
                plt.plot(fpr, tpr, label=filenames[gog])
            tt = len(posis)
            print(auc233)
            print('mean AUC', sum1 / tt)
            plt.legend(loc='lower right')
            # plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('False Positive Rate')  # 横坐标是fpr
            plt.ylabel('True Positive Rate')  # 纵坐标是tpr
            plt.title(u' ', FontProperties=font)
            # plt.figure(8,8)
            plt.savefig('nondeep.png')
            plt.show()
            # pdb.set_trace()
            # print(X_labels)
            # auc = roc_auc_score(X_labels, predict)
            # print(auc)
        myprob = "\n".join(map(str, predict))
        fw.write(myprob)
        fw.close()
    else:
        print('please specify that you want to train the mdoel or predict for your own sequences')


def parse_arguments(parser):

    parser.add_argument('--population_size', type=int, default=30,
                        help='The population size of evolutionary algorithms')
    parser.add_argument('--generation_number', type=int, default=10,
                        help='The generation number of evolutionary algorithms')

    parser.add_argument('--posi', type=str, metavar='<postive_sequecne_file>',
                        help='The fasta file of positive training samples')
    parser.add_argument('--nega', type=str, metavar='<negative_sequecne_file>',
                        help='The fasta file of negative training samples')
    parser.add_argument('--model_type', type=str, default='CNN',
                        help='it supports the following deep network models: CNN, CNN-LSTM, ResNet and DenseNet, default model is CNN')
    parser.add_argument('--out_file', type=str, default='prediction.txt',
                        help='The output file used to store the prediction probability of the testing sequences')
    parser.add_argument('--motif', type=bool, default=False,
                        help='It is used to identify binding motifs from sequences.')
    parser.add_argument('--motif_dir', type=str, default='motifs',
                        help='The dir used to store the prediction binding motifs.')
    parser.add_argument('--train', type=bool, default=True,
                        help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
    parser.add_argument('--model_file', type=str, default='model.pkl',
                        help='The file to save model parameters. Use this option if you want to train on your sequences or predict for your sequences')
    parser.add_argument('--predict', type=bool, default=False,
                        help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--testfile', type=str, default='',
                        help='the test fast file for sequences you want to predict for, you need specify it when using predict')
    parser.add_argument('--maxsize', type=int, default=501,
                        help='For global sequences, you need specify the maxmimum size to padding all sequences, it is only for global CNNs (default value: 501)')
    parser.add_argument('--channel', type=int, default=7,
                        help='The number of channels for breaking the entire RNA sequences to multiple subsequences, you can specify this value only for local CNNs (default value: 7)')
    parser.add_argument('--window_size', type=int, default=101,
                        help='The window size used to break the entire sequences when using local CNNs, eahc subsequence has this specified window size, default 101')
    parser.add_argument('--local', type=bool, default=False,
                        help='Only local multiple channel CNNs for local subsequences')
    parser.add_argument('--glob', type=bool, default=False,
                        help='Only global multiple channel CNNs for local subsequences')
    parser.add_argument('--ensemble', type=bool, default=True,
                        help='It runs the ensembling of local and global CNNs, you need specify the maxsize (default 501) for global CNNs and window_size (default: 101) for local CNNs')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='The number of filters for CNNs (default value: 16)')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of training epochs (default value: 50)')

    args = parser.parse_args()
    return args


parser = argparse.ArgumentParser()
args = parse_arguments(parser)
print(args)
# model_type = sys.argv[1]
run_edcnn(args)
