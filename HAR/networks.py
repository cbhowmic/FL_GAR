# import torch
import torch.nn as nn
# import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(561, 50)
        self.act1 = nn.ReLU()
        # self.hidden2 = nn.Linear(64, 32)
        # self.act2 = nn.ReLU()
        self.classification_layer = nn.Linear(50, 6)
        self.act3 = nn.LogSoftmax(dim=1)
        # self.tanh1 = nn.Tanh()
        # self.tanh2 = nn.Tanh()
        # self.act3 = nn.Sigmoid()

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        # X = self.hidden2(X)
        # X = self.act2(X)
        X = self.classification_layer(X)
        X = self.act3(X)
        return X


# def CREATE_MODEL(input_shape):
#     model = Sequential()
#     model.add(Input(shape = input_shape,))
#     model.add(Dense(256,activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(128,activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(64,activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(32,activation='relu'))
#     model.add(Dense(num_classes,activation='softmax'))
#     return model

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4 * 4 * 50, 500)
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


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

# class MLP(nn.Module):
#     def __int__(self):
#         super(MLP, self).__init__()
#         self.hidden = nn.Linear(561, 128)
#         self.classification_layer = nn.Linear(128, 6)
#         self.tanh1 = nn.Tanh()
#         self.tanh2 = nn.Tanh()
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         out = x.view(x.size(0), -1)
#         out = self.tanh1(out)
#         out = self.hidden(out)
#         out = self.tanh2(out)
#         out = self.classification_layer(out)
#         out = self.softmax(out)
#         return out

# class MLP(nn.Module):
#     def __init__(self,):
#         super(MLP, self).__init__()
#         self.hidden1 = nn.Linear(561, 128)
#         self.act1 = nn.ReLU()
#         self.hidden2 = nn.Linear(128, 64)
#         self.act2 = nn.ReLU()
#         self.classification_layer = nn.Linear(64, 6)
#         self.act3 = nn.LogSoftmax(dim=1)
#         # self.tanh1 = nn.Tanh()
#         # self.tanh2 = nn.Tanh()
#         # self.act3 = nn.Sigmoid()
#
#     def forward(self, X):
#         X = self.hidden1(X)
#         X = self.act1(X)
#         X = self.hidden2(X)
#         X = self.act2(X)
#         X = self.classification_layer(X)
#         X = self.act3(X)
#         return X


# class Linear(nn.Module):
#     def __init__(self):
#         super(Linear, self).__init__()
#         self.hidden1 = nn.Linear(561, 50)  # hidden layer
#         self.out = nn.Linear(50, 6)  # output layer
#
#     def forward(self, x):
#         # x = x.view(x.size(0), -1)
#         x = F.relu(self.hidden1(x))  # activation function for hidden layer
#         x = self.out(x)
#         return x


# class linearRegression(nn.Module):
#     def __init__(self, inputSize, outputSize):
#     # def __init__(self):
#         super(linearRegression, self).__init__()
#         self.linear = nn.Linear(inputSize, outputSize)
#
#     def forward(self, x):
#         out = self.linear(x)
#         return out
#
#
# def modelFactory(model):
#     if model == 'CNN':
#         return CNN
#     elif model == 'MLP':
#         return MLP
# #     elif model == 'LR':
# #         return linearRegression
# #     elif model == 'linear':
#         return Linear(n_feature=561, n_hidden1=50, n_output=6)