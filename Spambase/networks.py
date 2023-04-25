# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPclassification(nn.Module):
    def __init__(self):
        super(MLPclassification, self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(
                in_features=57,
                out_features=20,        # 20 originally
                bias=True,
            ),
            nn.ReLU()
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU()
        )

        self.classifica = nn.Sequential(
            nn.Linear(10, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        fc1 = self.hidden1(x.float())
        fc2 = self.hidden2(fc1)
        output = self.classifica(fc2)

        return fc1, fc2, output


class MLP(nn.Module):
    def __init__(self,):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(57, 64)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()
        self.classification_layer = nn.Linear(32, 2)
        self.act3 = nn.LogSoftmax(dim=1)
        # self.tanh1 = nn.Tanh()
        # self.tanh2 = nn.Tanh()
        # self.act3 = nn.Sigmoid()

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.classification_layer(X)
        X = self.act3(X)
        return X


# def modelFactory(model):
#     if model == 'CNN':
#         return CNN
#     elif model == 'MLP':
#         return MLP