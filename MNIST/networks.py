# import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __int__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.classification_layer = nn.Linear(hidden_size, output_size)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.tanh1(out)
        out = self.hidden(out)
        out = self.tanh2(out)
        out = self.classification_layer(out)
        out = self.softmax(out)
        return out


def modelFactory(model):
    if model == 'CNN':
        return CNN
    elif model == 'MLP':
        return MLP(input_size=784, hidden_size=50, output_size=10)