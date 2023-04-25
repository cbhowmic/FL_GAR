import syft as sy

from threading import Thread

from FLDataset import *
from aggregation import *
from networks import *
from operations import *
from config import *

" Step 2: specify the arguments needed for the experiment "
args = Arguments()

" Step 3: specify the device : CPU or GPU "
use_cuda = args.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

torch.manual_seed(1234)

" Step 4: define the hook and the workers"
hook = sy.TorchHook(torch)
clients = []
for i in range(args.clients):
    clients.append({'hook': sy.VirtualWorker(hook, id="client{}".format(i + 1))})

" Step 5: Get the dataset ready "
# Get the global and local datasets
if args.dataset == 'Spambase':
    global_train, global_test, train_group, test_group = load_SpamDataset(args.clients, args.iid)
    # global_train, global_test, train_group, test_group = load_SpamDataset(args)

print(len(global_train), len(global_test))
print(len(train_group[1]), len(test_group[1]))

# Load the global train and test set
global_train_loader = DataLoader(global_train, batch_size=args.global_batches, shuffle=False)
global_test_loader = DataLoader(global_test, batch_size=args.global_batches, shuffle=False)

# Load the local datasets at the workers
if args.attack and args.attack_type == 'label_flipping':
    for inx, client in enumerate(clients):
        if inx < args.num_attacked:
            flip_signal = True
        else:
            flip_signal = False
        trainset_ind_list = list(train_group[inx])
        client['trainset'] = get_worker_data(global_train, trainset_ind_list, args.local_batches, flip_signal)
        testset_ind_list = list(test_group[inx])
        client['testset'] = get_worker_data(global_test, testset_ind_list, args.local_batches, flip_signal)
else:
    for inx, client in enumerate(clients):
        trainset_ind_list = list(train_group[inx])
        client['trainset'] = get_worker_data(global_train, trainset_ind_list, args.local_batches, flip_signal=False)
        # print('client train data length:', len(client['trainset']))
        testset_ind_list = list(test_group[inx])
        client['testset'] = get_worker_data(global_test, testset_ind_list, args.local_batches, flip_signal=False)
        # print('client test data length:', len(client['testset']))


print('Experiment details: \n Num of clients:', args.clients, '\n Dataset:', args.dataset, '\n Network:', args.network, '\n Attack: ', args.attack, '\n Number of attacked:', args.num_attacked, '\n Attack type:', args.attack_type)


" Step 6: the main algorithm "
for rule in args.aggregate:
    print('============================\n')
    print('Aggregation rule : ', rule)

    "(1): Select the network "
    # Net = modelFactory(args.network)
    # Net = MLPclassification
    Net = MLP
    global_model = Net().to(device)
    grad_model = Net().to(device)
#
    "(2): define the optimizer function "
    optimizer = FedSGDOptim(global_model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(global_model.parameters(), lr=args.lr)

    "(3): set the seed, initialization"
    torch.manual_seed(args.torch_seed)
    # train_loss, train_accuracy, test_loss, test_accuracy = [], [], [], []
    global train_loss, train_accuracy, test_loss, test_accuracy
    train_loss = np.zeros((1, args.rounds))
    train_accuracy = np.zeros((1, args.rounds))
    test_loss = np.zeros((1, args.rounds))
    test_accuracy = np.zeros((1, args.rounds))
#
    "(4) define the client models and optimizer functions"
    for client in clients:
        torch.manual_seed(args.torch_seed)
        client['model'] = Net().to(device)
        # client['optim'] = torch.optim.Adam(client['model'].parameters(), lr=args.lr)
        client['optim'] = optim.SGD(client['model'].parameters(), lr=args.lr, momentum=0.9)

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

        # # Train the workers
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

        # Save the loss and accuracy at every comm round
        save_data(args, rule, train_loss, train_accuracy, test_loss, test_accuracy)

        # Share the global model with the clients
        for client in clients:
            client['model'].load_state_dict(global_model.state_dict())



# # Import necessary libraries
# import numpy as np
# import pandas as pd
# import torch
# import syft as sy
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# # Load Spambase dataset
# # data = pd.read_csv('spambase.data')
#
# data = pd.read_csv('spambase.data', names=[x for x in range(58)])
#
# # First let's shuffle the dataset
# data = data.sample(frac=1).reset_index(drop=True)
#
# # Now lets split dataset into features and target
# Y = data[57]
# del data[57]
# X = data
# # X = data.drop('Class', axis=1).values
# # y = data['Class'].values
#
# X = X.to_numpy()
# y = Y.to_numpy()
#
# # labels = torch.tensor(labels)
# # inputs = torch.tensor(inputs)
#
# # Split dataset into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Define the number of clients and batch size
# num_clients = 10
# batch_size = 32
#
# # Divide the dataset into clients
# client_data = []
# for i in range(num_clients):
#     start = i * (len(X_train) // num_clients)
#     end = (i+1) * (len(X_train) // num_clients)
#     client_data.append((torch.tensor(X_train[start:end]).float(), torch.tensor(y_train[start:end]).float()))
#
# # Initialize the global model weights
# global_model = torch.nn.Linear(X_train.shape[1], 1)
#
# # Define the number of federated rounds and the number of local epochs
# num_rounds = 10
# num_local_epochs = 10
#
# # Define the hook and workers for PySyft
# hook = sy.TorchHook(torch)
# workers = []
# for i in range(num_clients):
#     worker = sy.VirtualWorker(hook, id=f"worker{i+1}")
#     workers.append(worker)
#
# # Send the global model to all workers
# # global_model = global_model.send(workers)
# for worker in workers:
#     global_model = global_model.send(worker)
#
# # Train the models on each client and aggregate the gradients at each round
# for r in range(num_rounds):
#     print("Federated round:", r+1)
#     local_models = []
#     for i in range(num_clients):
#         model = global_model.copy().get()
#         optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
#         for e in range(num_local_epochs):
#             # Train the model on the client data
#             data, target = client_data[i]
#             data_ptr = data.send(workers[i])
#             target_ptr = target.send(workers[i])
#             model.train()
#             optimizer.zero_grad()
#             output = model(data_ptr)
#             loss = torch.nn.functional.binary_cross_entropy_with_logits(output.view(-1), target_ptr)
#             loss.backward()
#             optimizer.step()
#             # Get the updated model weights
#             model_ptr = model.send(workers[i])
#             local_models.append(model_ptr)
#
#     # Aggregate the gradients by averaging the weights of the local models
#     global_model = sy.utils.hook_args(global_model, "move")
#     global_model.move(sum(local_models).get() / (num_clients * num_local_epochs))
#
# # Get the final global model
# global_model = global_model.get()
#
# # Use the global model to make predictions on the test set
# global_model.eval()
# with torch.no_grad():
#     X_test_tensor = torch.tensor(X_test).float()
#     y_pred_tensor = global_model(X_test_tensor)
# y_pred = (y_pred_tensor > 0.5).float()
# final_preds = y_pred.view(-1).tolist()
#
# # Calculate the accuracy of the final predictions
# acc = accuracy_score(y_test, final_preds)
# print("Accuracy:", acc)
