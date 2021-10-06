import sys
import shutil, os
from shutil import copyfile

import sklearn
from sklearn import metrics
from collections import OrderedDict
import numpy as np
import random
import pdb
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils import class_weight
from deap.algorithms import varOr
from deap import base
from deap import creator
from deap import tools

gpu_id = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
mbest = 10
cxpb = 0.5
mutpb = 0.5
mu = 0.5
u = 10
lam = 60
ks = 1
kv = 1
ak = 0.9
bk = 1.1
sigma = []
random.seed(42)
num_epochs = 5
batch_size = 100
generation = 0

dataset_name = sys.argv[1]



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


class CNN(nn.Module):
    def __init__(self, nb_filter, channel=7, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), labcounts=32,
                 window_size=12, hidden_size=200, stride=(1, 1), padding=0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride=stride, padding=padding),
            # nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride=stride)
        out1_size = (window_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1

        maxpool_size = (out1_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1
        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size=(1, 10), stride=stride, padding=padding),
            # nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=stride))
        out2_size = (maxpool_size + 2 * padding - (kernel_size[1] - 1) - 1) / stride[1] + 1
        maxpool2_size = (out2_size + 2 * padding - (pool_size[1] - 1) - 1) / stride[1] + 1
        # self.drop1 = nn.Dropout(p=0.25)
        print('maxpool_size', maxpool_size)
        self.fc1 = nn.Linear(int(maxpool2_size * nb_filter), hidden_size)
        # self.fc1 = nn.Linear(7760, hidden_size)
        # self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        # out = self.drop1(out)
        out = self.fc1(out)
        # out = self.drop2(out)
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
        print('X.shape', X.shape)
        train_set = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                                  torch.from_numpy(y.astype(np.float32)).long().view(-1))
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        self.model.train()
        for t in range(nb_epoch):
            loss = self._fit(train_loader)
            print('loss', loss)

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


net1 = CNN(nb_filter=16, labcounts=4, window_size=507, channel=1)
net2 = CNN(nb_filter=16, labcounts=4, window_size=107, channel=7)


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
    for i in range(10):
        p.append(0.01 + 0.02 * (9 - i))
    p = np.array(p)
    index_p = []
    for i in range(10):
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


def fitscore(individual):
    return individual.fitness.values


def split_training_validation(classes, validation_size=0.2, shuffle=True):
    """split sampels based on balnace classes"""
    num_samples = len(classes)
    classes = np.array(classes)
    classes_unique = np.unique(classes)
    num_classes = len(classes_unique)
    indices = np.arange(num_samples)
    # indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl = indices[classes == cl]
        num_samples_cl = len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl)  # in-place shuffle

        # module and residual
        num_samples_each_split = int(num_samples_cl * validation_size)
        res = num_samples_cl - num_samples_each_split

        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res

        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl] * num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]

    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]

    return training_indice, training_label, validation_indice, validation_label


def read_fasta_file(fasta_file):
    seq_dict = {}
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        # distinguish header from sequence
        if line[0] == '>':  # or line.startswith('>')
            # it is the header
            name = line[1:]  # discarding the initial >
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()

    return seq_dict


def read_fasta_file_new(fasta_file='../data/UTR_hg19.fasta'):
    seq_dict = {}
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        # distinguish header from sequence
        if line[0] == '>':  # or line.startswith('>')
            # it is the header
            name = line[1:].split()[0]  # discarding the initial >
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()

    return seq_dict


def load_rnacomend_data(datadir='../data/'):
    pair_file = datadir + 'interactions_HT.txt'
    # rbp_seq_file = datadir + 'rbps_HT.fa'
    rna_seq_file = datadir + 'utrs.fa'

    rna_seq_dict = read_fasta_file(rna_seq_file)
    protein_set = set()
    inter_pair = {}
    new_pair = {}
    with open(pair_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split()
            protein = values[0]
            protein_set.add(protein)
            rna = values[1]
            inter_pair.setdefault(rna, []).append(protein)
            new_pair.setdefault(protein, []).append(rna)

    for protein, rna in new_pair.items():
        if len(rna) > 2000:
            print(protein)
    return inter_pair, rna_seq_dict, protein_set, new_pair


def er_get_all_rna_mildata(seqs, training_val_indice, test_indice):
    index = 0
    train_seqs = []
    for val in training_val_indice:
        train_seqs.append(seqs[val])

    test_seqs = []
    for val in test_indice:
        test_seqs.append(seqs[val])
    return train_seqs, test_seqs


criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    cuda = True
    # torch.cuda.set_device(1)
    print('===> Using GPU')
else:
    cuda = False
    print('===> Using CPU')


def padding_sequence_new(seq, max_len=101, repkey='N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < max_len:
        gap_len = max_len - seq_len
        new_seq = seq + repkey * gap_len
    return new_seq


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
            start = int((num_of_ins - channel) / 2)
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) < channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                # bag_subt.append(random.choice(bag_subt))
                tri_fea = get_RNA_seq_concolutional_array('N' * window_size)
                bag_subt.append(tri_fea.T)

        bags.append(np.array(bag_subt))

    return bags, labels


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


def padding_sequence(seq, max_len=501, repkey='N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len - seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq


def get_class_weight(df_y):
    y_classes = df_y.idxmax(1, skipna=False)

    from sklearn.preprocessing import LabelEncoder

    # Instantiate the label encoder
    le = LabelEncoder()

    # Fit the label encoder to our label series
    le.fit(list(y_classes))

    # Create integer based labels Series
    y_integers = le.transform(list(y_classes))

    # Create dict of labels : integer representation
    labels_and_integers = dict(zip(y_classes, y_integers))

    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    sample_weights = compute_sample_weight('balanced', y_integers)

    class_weights_dict = dict(zip(le.transform(list(le.classes_)), class_weights))

    return class_weights_dict


def get_all_rna_mildata(data, max_len=501):
    train_bags, train_labels = get_bag_data_1_channel(data, max_len)

    return train_bags, train_labels  # , test_bags, test_labels


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


def get_domain_features(in_file='rbps_HT.txt'):
    protein_list = []
    with open('protein_list', 'r') as fp:
        for line in fp:
            protein_list.append(line[1:-1])
    domain_dict = {}
    fp = open(in_file, 'r')
    index = 0
    for line in fp:
        values = line.rstrip().split()
        vals = [float(val) for val in values]
        domain_dict[protein_list[index]] = vals
        index = index + 1
    fp.close()

    return domain_dict


def runRBP47():
    inter_pair_dict, rna_seq_dict, protein_set, new_pair = load_rnacomend_data()
    protein_list = []
    for protein in protein_set:
        protein_list.append(protein)
    runEDCNN(inter_pair_dict, rna_seq_dict, protein_list, new_pair)


def train_network(model_type, X_train, y_train, channel=7, window_size=107, model_file='model.pkl', batch_size=100,
                  n_epochs=50, num_filters=16):
    print('model training for ', model_type)

    if model_type == 'CNN':
        model = CNN(nb_filter=num_filters, labcounts=4, window_size=window_size, channel=channel)

    # model.fc1 = nn.Linear(int(1360), 200)
    # model.fc2 = nn.Linear(200, 2)
    cwd = os.getcwd()
    path_ = cwd + '/' + model_file
    if os.path.isfile(path_):
        new_model = torch.load(model_file)
        model.load_state_dict(new_model)

    if cuda:
        model = model.cuda()
    clf = Estimator(model)
    rand = random.random()
    lr1 = random.uniform(0.01, 0.1)
    lr2 = random.uniform(1e-4, 1e-3)
    lr3 = random.uniform(1e-5, 1e-4)
    mt = random.uniform(0.9, 0.99)
    wd = random.uniform(1e-6, 1e-5)
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
    torch.save(model.state_dict(), model_file)
    model = model.cpu()
    torch.cuda.empty_cache()
    return model


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


def runEDCNN(inter_pair_dict, rna_seq_dict, protein_list, new_pair):
    max_num_targets = 0
    batch_size = 100
    n_epochs = 20
    num_filters = 16
    window_size = 101
    max_size = 501
    model_type = 'CNN'
    data = {}
    labels = []
    rna_seqs = []
    protein_list.append("negative")
    # tt = 0
    all_hg19_utrs = read_fasta_file_new()
    # for i in inter_pair_dict.keys():
    #     if i not in all_hg19_utrs.keys():
    #         tt = tt + 1
    # print('142424tt', tt)
    print('inter_pair_dict.keys()', len(inter_pair_dict.keys()))
    print('all_hg19_utrs.keys()', len(all_hg19_utrs.keys()))
    remained_rnas = list(set(all_hg19_utrs.keys()) - set(inter_pair_dict.keys()))
    # pdb.set_trace()
    flag = 0
    for protein1, rna1 in new_pair.items():
        print(protein1, len(rna1))
        # print(sys.argv[1])
        if protein1 == dataset_name:
            flag = 1
            print(protein1)
            max_num_targets = len(rna1)
            for i in range(len(rna1)):
                rna_seq = rna_seq_dict[rna1[i]]
                rna_seq = rna_seq.replace('T', 'U')
                rna_seqs.append(rna_seq)
                labels.append(1)
        else:
            continue
        if flag:
            break
    random.shuffle(remained_rnas)
    for rna in remained_rnas[:max_num_targets]:
        rna_seq = all_hg19_utrs[rna]
        rna_seq = rna_seq.replace('T', 'U')
        rna_seqs.append(rna_seq)
        labels.append(0)

    # x_index = range(len(labels))
    print(len(rna_seqs))
    print(len(labels))
    train_seqs, test_seqs, train_label, test_label = train_test_split(rna_seqs, labels, random_state=42, shuffle=True,
                                                                      test_size=0.2)
    print('len(train_seqs)', len(train_seqs))
    print('len(test_seqs)', len(test_seqs))
    train_seqs, val_seqs, train_label, val_label = train_test_split(train_seqs, train_label, random_state=42, shuffle=True,
                                                                      test_size=0.1)
    data = dict()
    data['seq'] = train_seqs
    data['Y'] = train_label
    train_bags1, train_labels1 = get_bag_data_1_channel(data, max_len=501)
    train_bags2, train_labels2 = get_bag_data(data, channel=7, window_size=101)
    data_val = dict()
    data_val['seq'] = val_seqs
    data_val['Y'] = val_label
    val_bags1, val_labels1 = get_bag_data_1_channel(data_val, max_len=501)
    val_bags2, val_labels2 = get_bag_data(data_val, channel=7, window_size=101)

    model_file = dataset_name
    population = toolbox.population(u)
    for k in range(20):
        generation = k
        print('generation %d' % (k))
        for j in range(u):
            print('individual:%d' % (j))
            model_1 = CNN(nb_filter=num_filters, labcounts=4, window_size=window_size + 6,
                          channel=1)
            model_1 = train_network('CNN', np.array(train_bags1), np.array(train_labels1),
                                    channel=1,
                                    window_size=501 + 6,
                                    model_file=str(
                                        j) + '.' + model_file + '.global',
                                    batch_size=100,
                                    n_epochs=5,
                                    num_filters=16)
            model_2 = CNN(nb_filter=num_filters, labcounts=4, window_size=window_size + 6,
                          channel=7)
            model_2 = train_network('CNN', np.array(train_bags2), np.array(train_labels2),
                                    channel=7,
                                    window_size=101 + 6,
                                    model_file=str(
                                        j) + '.' + model_file + '.local',
                                    batch_size=100,
                                    n_epochs=5,
                                    num_filters=16)
            population[j][0] = model_1.state_dict()
            population[j][1] = model_2.state_dict()
            torch.cuda.empty_cache()
        for v in range(kv):
            for ind in population:
                predict1 = individual_evaluation(model_type, np.array(val_bags1), channel=1,
                                             window_size=max_size + 6,
                                             individual=ind[0],
                                             batch_size=batch_size,
                                             n_epochs=n_epochs,
                                             num_filters=num_filters)
                predict2 = individual_evaluation(model_type, np.array(val_bags2), channel=7,
                                             window_size=window_size + 6,
                                             individual=ind[1],
                                             batch_size=batch_size,
                                             n_epochs=n_epochs,
                                             num_filters=num_filters)
                predict = (predict1 + predict2) / 2.0
                auc = roc_auc_score(val_labels2, predict)
                gg = []
                gg.append(auc)
                ind.fitness.values = gg
                torch.cuda.empty_cache()
            population.sort(key=fitscore, reverse=True)
            offspring = generate_offspring(population, toolbox, 4 * u, cxpb, mutpb, k)
            for ind in offspring:
                predict1 = individual_evaluation(model_type, np.array(val_bags1), channel=1,
                                             window_size=max_size + 6,
                                             individual=ind[0],
                                             batch_size=batch_size,
                                             n_epochs=n_epochs,
                                             num_filters=num_filters)
                predict2 = individual_evaluation(model_type, np.array(val_bags2), channel=7,
                                             window_size=window_size + 6,
                                             individual=ind[1],
                                             batch_size=batch_size,
                                             n_epochs=n_epochs,
                                             num_filters=num_filters)
                predict = (predict1 + predict2) / 2.0
                auc = roc_auc_score(val_labels2, predict)
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
                torch.save(ind[0], str(
                    cnt) + '.' + model_file + '.global')
                torch.save(ind[1], str(
                    cnt) + '.' + model_file + '.local')
                cnt = cnt + 1
            torch.cuda.empty_cache()
        with torch.no_grad():
            data2 = dict()
            data2['seq'] = test_seqs
            data2['Y'] = test_label
            test_bags1, test_labels1 = get_bag_data_1_channel(data2, max_len=501)
            test_bags2, test_labels2 = get_bag_data(data2, channel=7, window_size=101)
            auclist = []
            best_score = 0
            for j in range(u):
                predict1 = predict_network(model_type, np.array(test_bags1), channel=1, window_size=max_size + 6,
                                           model_file=str(
                                               j) + '.' + model_file + '.global', batch_size=batch_size,
                                           n_epochs=n_epochs,
                                           num_filters=num_filters)
                predict2 = predict_network(model_type, np.array(test_bags2), channel=7, window_size=window_size + 6,
                                           model_file=str(
                                               j) + '.' + model_file + '.local', batch_size=batch_size,
                                           n_epochs=n_epochs,
                                           num_filters=num_filters)
                predict = (predict1 + predict2) / 2.0
                auc = roc_auc_score(test_labels2, predict)
                auclist.append(auc)
                if auc > best_score:
                    best_score = auc
                    copyfile(str(j) + '.' + model_file + '.global',
                             'best' + '.' + model_file + '.global')
                    copyfile(str(j) + '.' + model_file + '.local',
                             'best' + '.' + model_file + '.local')
                torch.cuda.empty_cache()
            auclist.sort(reverse=True)
            f1 = open(dataset_name + '.txt', 'w')
            for i in auclist:
                f1.write('AUC:' + str(i) + '\n')
            f1.write('end')
            f1.close()
            gc.collect()
            torch.cuda.empty_cache()

    del train_bags1, train_labels1
    del train_bags2, train_labels2
    gc.collect()
    torch.cuda.empty_cache()


def run_edcnn():
    runRBP47()


run_edcnn()

