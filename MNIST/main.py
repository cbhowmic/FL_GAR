""" Federated Learning repository """
# Author: Chandreyee Bhowmick
# Date: November 2022
# References used: (1) https://github.com/Gharibim/federated_learning_course/blob/main/Federated_Learning_techniques/FedSGD.ipynb
#                  (2) Byrd_SAGA

" Step 1: Import the relevant packages and modules "
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Dataset
import syft as sy
# import copy
# import numpy as np
# import time
# import os
# import importlib
from threading import Thread

from FLDataset import *
# from utils import *
from aggregation import *
# from plot import *
from networks import *
from operations import *
from config import *

" Step 2: specify the arguments needed for the experiment "
args = Arguments()

" Step 3: specify the device : CPU or GPU "
use_cuda = args.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

" Step 4: define the hook and the workers"
hook = sy.TorchHook(torch)
clients = []
for i in range(args.clients):
    clients.append({'hook': sy.VirtualWorker(hook, id="client{}".format(i + 1))})

" Step 5: Get the dataset ready "
# Get the global and local datasets
if args.dataset == 'MNIST':
    global_train, global_test, train_group, test_group = load_MNISTdataset(args.clients, args.iid)
# print('type', type(global_train), len(global_train))

# Load the local datasets at the workers
if args.attack and args.attack_type == 'label_flipping':
    for inx, client in enumerate(clients):
        if inx < args.num_attacked:
            flip_signal = False
        else:
            flip_signal = False
        # print('inx:', inx, 'flip', flip_signal)
        trainset_ind_list = list(train_group[inx])
        client['trainset'] = get_images(global_train, trainset_ind_list, args.local_batches, flip_signal)
        client['testset'] = get_images(global_test, list(test_group[inx]), args.local_batches, flip_signal)
        client['samples'] = len(trainset_ind_list) / args.images
else:
    for inx, client in enumerate(clients):
        trainset_ind_list = list(train_group[inx])
        client['trainset'] = get_images(global_train, trainset_ind_list, args.local_batches, flip_signal=False)
        client['testset'] = get_images(global_test, list(test_group[inx]), args.local_batches, flip_signal=False)
        client['samples'] = len(trainset_ind_list) / args.images
        # print('client dataset', type(client['trainset']), len(client['trainset']))
# Load the global train and test set
global_train_loader = DataLoader(global_train, batch_size=args.local_batches, shuffle=False)   # shuffle=True
global_test_loader = DataLoader(global_test, batch_size=args.local_batches, shuffle=False)     # shuffle=True
# print('type of loader', type(global_train_loader), 'len', len(global_train_loader))

print('Experiment details: \n Num of clients:', args.clients, '\n Dataset:', args.dataset, '\n Network:', args.network,
      '\n Attack: ', args.attack, '\n Number of attacked:', args.num_attacked, '\n Attack type:', args.attack_type)

" Step 6: the main algorithm "
for rule in args.aggregate:
    print('============================\n')
    print('Aggregation rule : ', rule)

    "(1): Select the network "
    Net = modelFactory(args.network)
    global_model = Net().to(device)
    grad_model = Net().to(device)

    "(2): define the optimizer function "
    optimizer = FedSGDOptim(global_model.parameters(), lr=args.lr)

    "(3): set the seed, initialization"
    torch.manual_seed(args.torch_seed)
    # train_loss, train_accuracy, test_loss, test_accuracy = [], [], [], []
    global train_loss, train_accuracy, test_loss, test_accuracy
    train_loss = np.zeros((1, args.rounds))
    train_accuracy = np.zeros((1, args.rounds))
    test_loss = np.zeros((1, args.rounds))
    test_accuracy = np.zeros((1, args.rounds))

    "(4) define the client models and optimizer functions"
    for client in clients:
        torch.manual_seed(args.torch_seed)
        client['model'] = Net().to(device)
        client['optim'] = optim.SGD(client['model'].parameters(), lr=args.lr)

    # t0 = time.time()
    "(5) federated communication round"
    for fed_round in range(args.rounds):
        print('comm round:', fed_round)

        # number of selected clients
        m = int(max(args.C * args.clients, 1))

        # Selected devices
        np.random.seed(fed_round)
        selected_clients_inds = np.random.choice(range(len(clients)), m, replace=False)
        selected_clients = [clients[i] for i in selected_clients_inds]

        # Active devices
        np.random.seed(fed_round)
        active_clients_inds = np.random.choice(selected_clients_inds, int((1 - args.drop_rate) * m), replace=False)
        active_clients = [clients[i] for i in active_clients_inds]

        # Train the workers
        # client_loss = 0
        # for client in active_clients:
        #     train(args, client, device)

        # Train the workers in parallel
        client_loss = 0
        Threads = [Thread(target=train(args, client, device)) for client in active_clients]
        for t_k in Threads:
            t_k.start()
        for t_k in Threads:
            t_k.join()

        # Aggregate the gradients from the workers
        global_model = aggregate_gradients(global_model, active_clients, args, rule)
        optimizer.step(grad_model)

        # Calculate the loss and accuracy
        train_l, train_acc = loss_and_accuracy(global_model, device, global_train_loader, 'Train')
        test_l, test_acc = loss_and_accuracy(global_model, device, global_test_loader, 'Test')
        train_loss[0, fed_round] = train_l
        train_accuracy[0, fed_round] = train_acc
        test_loss[0, fed_round] = test_l
        test_accuracy[0, fed_round] = test_acc
        # train_loss.append(train_l)
        # test_loss.append(test_l)
        # train_accuracy.append(100 * train_acc)
        # test_accuracy.append(100 * test_acc)

        # Save the loss and accuracy at every comm round
        save_data(args, rule, train_loss, train_accuracy, test_loss, test_accuracy)

        # Share the global model with the clients
        for client in clients:
            client['model'].load_state_dict(global_model.state_dict())

        # print('time taken in the communication round ', fed_round, 'is ', time.time() - t0)
        # t0 = time.time()

    "(6) Save the trained global model"
    if args.save_model:
        save_model(args, rule, global_model)

    # "(7) Save the loss and accuracy data"
    # if args.save_data:
    #     save_data(args, rule, train_loss, train_accuracy, test_loss, test_accuracy)

# torch.manual_seed(args.torch_seed)
# train_loss, train_accuracy, test_loss, test_accuracy = [], [], [], []
#
# # define the client models and optimizer functions
# for client in clients:
#     torch.manual_seed(args.torch_seed)
#     client['model'] = Net().to(device)
#     client['optim'] = optim.SGD(client['model'].parameters(), lr=args.lr)
#
# # federated communication round
# for fed_round in range(args.rounds + 1):
#     print('comm round:', fed_round)
#     # number of selected clients
#     m = int(max(args.C * args.clients, 1))
#     # Selected devices
#     np.random.seed(fed_round)
#     selected_clients_inds = np.random.choice(range(len(clients)), m, replace=False)
#     selected_clients = [clients[i] for i in selected_clients_inds]
#     # Active devices
#     np.random.seed(fed_round)
#     active_clients_inds = np.random.choice(selected_clients_inds, int((1 - args.drop_rate) * m), replace=False)
#     active_clients = [clients[i] for i in active_clients_inds]
#
#     # Train the workers
#     client_loss = 0
#     for client in active_clients:
#         train(args, client, device)
#
#     # Aggregate the gradients from the workers
#     global_model = aggregate_gradients(global_model, active_clients, args)
#     # print('global', type(global_model))
#     optimizer.step(grad_model)
#
#     # Calculate the loss and accuracy
#     train_l, train_acc = loss_and_accuracy(global_model, device, global_train_loader, 'Train')
#     test_l, test_acc = loss_and_accuracy(global_model, device, global_test_loader, 'Test')
#     train_loss.append(train_l)
#     test_loss.append(test_l)
#     train_accuracy.append(100 * train_acc)
#     test_accuracy.append(100 * test_acc)
#
#     # Share the global model with the clients
#     for client in clients:
#         client['model'].load_state_dict(global_model.state_dict())
#
# if args.save_model:
#     save_model(args, global_model)
#     # attack_str = (args.attack_type + "attacked" + args.num_attacked) if args.attack else ""
#     # torch.save(global_model.state_dict(), "models/Global_%s_%s_%s_%sclients_%s_%s.pt" % (args.dataset, args.iid, args.network, args.clients, args.aggregate, attack_str))
#
# if args.save_data:
#     save_data(args, train_loss, train_accuracy, test_loss, test_accuracy)