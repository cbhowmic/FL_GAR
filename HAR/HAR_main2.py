""" Federated Learning repository : Human Activity Recognition module"""
# Author: Chandreyee Bhowmick
# Date: April 2023
# References used: (1) https://github.com/Gharibim/federated_learning_course/blob/main/Federated_Learning_techniques/FedSGD.ipynb
# (2) Byrd_SAGA, (3) https://github.com/JianiLi/resilientDistributedMTL/tree/main/HumanActivityRecog,
# (4) https://github.com/MadhavShashi/Human-Activity-Recognition-Using-Smartphones-Sensor-DataSet/blob/master/1.HumanActivityRecognition_EDA.ipynb

" Step 1: Import the relevant packages and modules "
import syft as sy
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


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


" Step 3: specify the device : CPU or GPU "
device = get_device()
kwargs = {'num_workers': 1, 'pin_memory': True}
torch.manual_seed(1234)


" Step 4: define the hook and the workers"
hook = sy.TorchHook(torch)
clients = []
for i in range(args.clients):
    clients.append({'hook': sy.VirtualWorker(hook, id="client{}".format(i + 1))})


" Step 5: Get the dataset ready "
# def readData():
#
#     features = pd.read_csv('./UCI HAR Dataset/features.txt', sep='\s+', index_col=0, header=None)
#     # train_data = pd.read_csv('./UCI HAR Dataset/train/X_train.txt', sep='\s+',
#     #                          names=list(features.values.ravel()))
#     train_data = pd.read_csv('UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
#     # test_data = pd.read_csv('./UCI HAR Dataset/test/X_test.txt', sep='\s+',
#     #                         names=list(features.values.ravel()))
#     test_data = pd.read_csv('UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)
#
#     train_label = pd.read_csv('./UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None)
#     test_label = pd.read_csv('./UCI HAR Dataset/test/y_test.txt', sep='\s+', header=None)
#
#     train_subject = pd.read_csv('./UCI HAR Dataset/train/subject_train.txt', sep='\s+', header=None)
#     test_subject = pd.read_csv('./UCI HAR Dataset/test/subject_test.txt', sep='\s+', header=None)
#
#     label_name = pd.read_csv('UCI HAR Dataset/activity_labels.txt', sep='\s+', header=None, index_col=0)
#
#     train_data['label'] = train_label
#     test_data['label'] = test_label
#
#     train_data['subject'] = train_subject
#     test_data['subject'] = test_subject
#
#     def get_label_name(num):
#         return label_name.iloc[num - 1, 0]
#
#     train_data['label_name'] = train_data['label'].map(get_label_name)
#     test_data['label_name'] = test_data['label'].map(get_label_name)
#
#     # 原来标签为1-6，而算法需要0-5
#     train_data['label'] = train_data['label'] - 1
#     test_data['label'] = test_data['label'] - 1
#
#     np.random.shuffle(train_data.values)
#     np.random.shuffle(test_data.values)
#
#     return train_data, test_data


# def generateData(train_data, test_data, subject, batch_size, flip_signal):
#     x_train = [d[:-3] for d in train_data.values if d[-2] == subject]
#     y_train = [d[-3] for d in train_data.values if d[-2] == subject]
#     x_test = [d[:-3] for d in test_data.values if d[-2] == subject]
#     y_test = [d[-3] for d in test_data.values if d[-2] == subject]
#
#     all_x_data = x_train + x_test
#     all_y_data = y_train + y_test
#     # print('y data before attack:', type(all_y_data), len(all_y_data), min(all_y_data), max(all_y_data))
#
#     if flip_signal:
#         for i in range(len(all_y_data)):
#             all_y_data[i] = 5 - all_y_data[i]
#     # print('y data after attack', type(all_y_data), len(all_y_data), min(all_y_data), max(all_y_data))
#
#     x_tensor = torch.FloatTensor(all_x_data)
#     y_tensor = torch.LongTensor(all_y_data)
#
#     all_data = []
#     for i in range(len(x_tensor)):
#         all_data.append([x_tensor[i], y_tensor[i]])
#
#     np.random.shuffle(all_data)
#
#     train_data_subject, val_data_subject, test_data_subject = all_data[:len(all_data) // 4 * 3], \
#                                                               all_data[len(all_data) // 4 * 3: len(all_data) // 8 * 7], all_data[len(all_data) // 4 * 3:]
#
#     # x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = x_tensor[:len(x_tensor) // 4 * 3], y_tensor[:len(x_tensor) // 4 * 3], \
#     #                                    x_tensor[len(x_tensor) // 4 * 3:], y_tensor[len(x_tensor) // 4 * 3:]
#     # x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor = x_test_tensor[:len(x_test_tensor) // 2], y_test_tensor[:len(x_test_tensor) // 2], \
#     #                                     x_test_tensor[len(x_test_tensor) // 2:], y_test_tensor[len(x_test_tensor) // 2:]
#
#     # train_data_subject = []
#     # for i in range(len(x_train_tensor)):
#     #     train_data_subject.append([x_train_tensor[i], y_train_tensor[i]])
#     #
#     # val_data_subject = []
#     # for i in range(len(x_val_tensor)):
#     #    val_data_subject.append([x_val_tensor[i], y_val_tensor[i]])
#     #
#     # test_data_subject = []
#     # for i in range(len(x_test_tensor)):
#     #    test_data_subject.append([x_test_tensor[i], y_test_tensor[i]])
#
#     # un-even data
#     # if subject % 5 == 0:
#     #     train_data_subject, val_data_subject, test_data_subject = train_data_subject[:len(train_data_subject)//10], \
#     #                                     val_data_subject[:len(val_data_subject)//10], test_data_subject[:len(test_data_subject)//10]
#
#     train_loader_subject = DataLoader(dataset=train_data_subject, batch_size=batch_size, shuffle=True)
#     test_loader_subject = DataLoader(dataset=test_data_subject, batch_size=batch_size, shuffle=True)
#     val_loader_subject = DataLoader(dataset=val_data_subject, batch_size=batch_size, shuffle=True)
#
#     return train_loader_subject, val_loader_subject, test_loader_subject


# def dataProcess(data):
#     x_data = [d[:-3] for d in data.values]
#     y_data = [d[-3] for d in data.values]
#
#     x_tensor = torch.FloatTensor(x_data)
#     y_tensor = torch.LongTensor(y_data)
#
#     data_tensor = []
#     for i in range(len(x_tensor)):
#         data_tensor.append([x_tensor[i], y_tensor[i]])
#     np.random.shuffle(data_tensor)
#
#     return data_tensor

global_train, global_test = readData()
global_train_data = dataProcess(global_train)
global_test_data = dataProcess(global_test)
# Load the global train and test set
global_train_loader = DataLoader(global_train_data, batch_size=args.local_batches, shuffle=False)
global_test_loader = DataLoader(global_test_data, batch_size=args.local_batches, shuffle=False)

# Load the local datasets at the workers
if args.attack and args.attack_type == 'label_flipping':
    for inx, client in enumerate(clients):
        if inx < args.num_attacked:
            flip_signal = True
        else:
            flip_signal = False
        client['trainset'], client['validset'], client['testset'] = generateData(global_train, global_test, inx + 1,
                                                                                 args.local_batches, flip_signal)
        client['samples'] = len(client['trainset']) / args.total_rows
else:
    for inx, client in enumerate(clients):
        client['trainset'], client['validset'], client['testset'] = generateData(global_train, global_test, inx + 1, args.local_batches, flip_signal = False)
        client['samples'] = len(client['trainset']) / args.total_rows


" Step 6: the main algorithm "
for rule in args.aggregate:
    print('============================\n')
    print('Aggregation rule : ', rule)

    "(1): Select the network "
    Net = MLP
    global_model = Net()
    grad_model = Net()

    "(2): define the optimizer function "
    optimizer = FedSGDOptim(global_model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(global_model.parameters(), lr=args.lr)

    "(3): set the seed, initialization"
    torch.manual_seed(args.torch_seed)
    global train_loss, train_accuracy, test_loss, test_accuracy
    train_loss = np.zeros((1, args.rounds))
    train_accuracy = np.zeros((1, args.rounds))
    test_loss = np.zeros((1, args.rounds))
    test_accuracy = np.zeros((1, args.rounds))

    "(4) define the client models and optimizer functions"
    for client in clients:
        torch.manual_seed(args.torch_seed)
        client['model'] = Net()
        # client['optim'] = torch.optim.Adam(client['model'].parameters(), lr=args.lr)
        client['optim'] = optim.SGD(client['model'].parameters(), lr=args.lr, momentum=0.9)

    "(5) federated communication round"
    last_accuracy = 0
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

        # Train the workers in parallel
        client_loss = 0
        Threads = [Thread(target=train(args, client, 'cpu')) for client in active_clients]
        for t_k in Threads:
            t_k.start()
        for t_k in Threads:
            t_k.join()

        # Aggregate the gradients from the workers
        global_model = aggregate_gradients(global_model, active_clients, args, rule)
        optimizer.step(grad_model)

        # Calculate the loss and accuracy
        device = 'cpu'
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

        # Save the global model if it improves the testing accuracy
        if test_acc > last_accuracy:
            last_accuracy = test_acc
            # print('Saving a better model..')
            save_model(args, rule, global_model)
