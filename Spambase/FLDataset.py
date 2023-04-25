import torch
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
# import syft as sy


def spambaseIID(dataset, num_users):
    data_per_worker = int(len(dataset)/num_users)
    users_dict, indeces = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(i)
        users_dict[i] = set(np.random.choice(indeces, data_per_worker, replace=False))
        indeces = list(set(indeces) - users_dict[i])
    return users_dict


def load_SpamDataset(num_users, iidtype):

    # Loading dataset into pandas dataframe
    data = pd.read_csv('spambase.data', names=[x for x in range(58)])

    # First let's shuffle the dataset
    data = data.sample(frac=1).reset_index(drop=True)

    # Now lets split dataset into features and target
    Y = data[57]
    del data[57]
    X = data

    inputs = X.to_numpy()
    labels = Y.to_numpy()

    labels = torch.tensor(labels)
    inputs = torch.tensor(inputs)

    # splitting training and test data
    pct_test = 0.25

    train_labels = labels[:-int(len(labels) * pct_test)]
    train_inputs = inputs[:-int(len(labels) * pct_test)]

    test_labels = labels[-int(len(labels) * pct_test):]
    test_inputs = inputs[-int(len(labels) * pct_test):]

    # train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2)
    # print(len(train_labels), len(test_labels))

    global_train, global_test = [], []

    for i in range(len(train_inputs)):
        global_train.append([train_inputs[i], train_labels[i]])
    for i in range(len(test_inputs)):
        global_test.append([test_inputs[i], test_labels[i]])

    if iidtype == 'iid':
        train_group = spambaseIID(global_train, num_users)
        test_group = spambaseIID(global_test, num_users)
    # np.random.shuffle(global_train)
    # np.random.shuffle(global_test)
    # global_train = sy.BaseDataset(train_inputs, train_labels)
    # global_test = sy.BaseDataset(test_inputs, test_labels)

    return global_train, global_test, train_group, test_group

class FedDataset(Dataset):
    def __init__(self, dataset, indx, flip):
        self.dataset = dataset
        self.indx = [int(i) for i in indx]
        self.label_flip = flip

    def __len__(self):
        return len(self.indx)

    def __getitem__(self, item):
        images, label = self.dataset[self.indx[item]]
        # print('label before attack:', label, type(label))
        if self.label_flip == True:
            # print('label before attack:', label, type(label))
            # print('changing label')
            label = 1 - label
            # print('label after attack:', label)
        return torch.tensor(images).clone().detach(), torch.tensor(label).clone().detach()

def get_worker_data(dataset, indeces, batch_size, flip_signal):
    return DataLoader(FedDataset(dataset, indeces, flip=flip_signal), batch_size=batch_size, shuffle=True)