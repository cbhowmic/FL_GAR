import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import  numpy as np
import pandas as pd


# def HAR_iid(dataset, num_users):
#     data_per_worker = int(len(dataset) / num_users)
#     users_dict, indeces = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         np.random.seed(i)
#         users_dict[i] = set(np.random.choice(indeces, data_per_worker, replace=False))
#         indeces = list(set(indeces) - users_dict[i])
#     # return users_dict


# def read_HARdataset(num_users, iidtype):
#     features = list()
#     with open('UCI HAR Dataset/features.txt') as f:
#         features = [line.split()[1] for line in f.readlines()]
#     print('No of Features: {}'.format(len(features)))
#     X_train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
#     X_train.columns = [features]
#
#     X_train['subject'] = pd.read_csv('UCI HAR Dataset/train/subject_train.txt', header=None).squeeze("columns")
#     y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', names=['Activity'], ).squeeze("columns")
#     y_train_labels = y_train.map(
#         {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', 4: 'SITTING', 5: 'STANDING', 6: 'LAYING'})
#
#     # put all columns in a single dataframe
#     train = X_train
#     train['label'] = y_train
#     train['label_name'] = y_train_labels
#
#     # get the data from txt files to pandas dataffame
#     X_test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)
#     X_test.columns = [features]
#     # add subject column to the dataframe
#     X_test['subject'] = pd.read_csv('UCI HAR Dataset/test/subject_test.txt', header=None, ).squeeze("columns")
#
#     # get y labels from the txt file
#     y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', names=['Activity'], ).squeeze("columns")
#     y_test_labels = y_test.map({1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', \
#                                 4: 'SITTING', 5: 'STANDING', 6: 'LAYING'})
#
#     # put all columns in a single dataframe
#     test = X_test
#     test['label'] = y_test
#     test['label_name'] = y_test_labels
#
#     x_train = [d[:-3] for d in train.values]
#     y_train = [d[-3] for d in train.values]
#     x_test = [d[:-3] for d in test.values]
#     y_test = [d[-3] for d in test.values]
#
#     train_x_tensor = torch.FloatTensor(x_train)
#     train_y_tensor = torch.FloatTensor(y_train)
#     test_x_tensor = torch.FloatTensor(x_test)
#     test_y_tensor = torch.FloatTensor(y_test)
#
#     train_data, test_data = [], []
#
#     for i in range(len(train_x_tensor)):
#         train_data.append([train_x_tensor[i], train_y_tensor[i]])
#     for i in range(len(test_x_tensor)):
#         test_data.append([test_x_tensor[i], test_y_tensor[i]])
#
#     # train_group, test_group = None, None
#     # if iidtype == 'iid':
#     #     train_group = HAR_iid(train, num_users)
#     #     test_group = HAR_iid(test, num_users)
#     # else:
#     #     raise Exception("Non-iid distribution of HAR dataset is not implemented!")
#     return train, test, train_data, test_data
#
#
#     # # get the features from the file features.txt
#     # features = list()
#     # with open('UCI HAR Dataset/features.txt') as f:
#     #     features = [line.split()[1] for line in f.readlines()]
#     # print('No of Features: {}'.format(len(features)))
#     #
#     # # get the data from txt files to pandas dataffame
#     # X_train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
#     # X_train.columns = [features]
#     #
#     # # add subject column to the dataframe
#     # X_train['subject'] = pd.read_csv('UCI HAR Dataset/train/subject_train.txt', header=None).squeeze("columns")
#     #
#     # y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', names=['Activity'], ).squeeze("columns")
#     # y_train_labels = y_train.map(
#     #     {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', 4: 'SITTING', 5: 'STANDING', 6: 'LAYING'})
#     #
#     # # put all columns in a single dataframe
#     # train = X_train
#     # train['Activity'] = y_train
#     # train['ActivityName'] = y_train_labels
#     #
#     # # get the data from txt files to pandas dataffame
#     # X_test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)
#     # X_test.columns = [features]
#     # # add subject column to the dataframe
#     # X_test['subject'] = pd.read_csv('UCI HAR Dataset/test/subject_test.txt', header=None,).squeeze("columns")
#     #
#     # # get y labels from the txt file
#     # y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', names=['Activity'],).squeeze("columns")
#     # y_test_labels = y_test.map({1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', \
#     #                             4: 'SITTING', 5: 'STANDING', 6: 'LAYING'})
#     #
#     # # put all columns in a single dataframe
#     # test = X_test
#     # test['Activity'] = y_test
#     # test['ActivityName'] = y_test_labels
#     #
#     # train_group, test_group = None, None
#     # if iidtype == 'iid':
#     #     train_group = HAR_iid(train, num_users)
#     #     test_group = HAR_iid(test, num_users)
#     # else:
#     #     raise Exception("Non-iid distribution of HAR dataset is not implemented!")
#     # return train, test, train_group, test_group

# def local_HAR_data(train_data, test_data, clients):
#     train_group, test_group = None, None
#     images = int(len(dataset) / num_users)
#     users_dict, indeces = {}, [i for i in range(len(dataset))]

# def local_HARdata(train_data, test_data, subject, flip_signal, batch_size):
#
#     # x_train = train_data[train_data['subject'] == str(ind+1)]
#     # # # print('jgsfg', train_data.columns)
#     # # x_train = [d[:-3] for d in train_data.values]
#     # # y_train = [d[-3] for d in train_data.values]
#     # # x_test = [d[:-3] for d in test_data.values]
#     # # y_test = [d[-3] for d in test_data.values]
#     x_train = [d[:-3] for d in train_data.values if d[-2] == subject]
#     y_train = [d[-3] for d in train_data.values if d[-2] == subject]
#     x_test = [d[:-3] for d in test_data.values if d[-2] == subject]
#     y_test = [d[-3] for d in test_data.values if d[-2] == subject]
#
#     #
#     # print('index' ,ind, 'len1', len(x_train), '2', len(x_test), '3', len(y_train), '4', len(y_test))
#     #
#     # train_x_tensor = torch.FloatTensor(x_train)
#     # train_y_tensor = torch.LongTensor(y_train)
#     # test_x_tensor = torch.FloatTensor(x_test)
#     # test_y_tensor = torch.LongTensor(y_test)
#     # #
#     # train_data, test_data = [], []
#     # #
#     # # # print('tensor length', len(train_x_tensor))
#     # #
#     # if flip_signal:
#     #     for i in range(len(train_x_tensor)):
#     #         train_data.append([train_x_tensor[i], (7 - train_y_tensor[i])])
#     #     for i in range(len(test_x_tensor)):
#     #         test_data.append([test_x_tensor[i], test_y_tensor[i]])
#     # else:
#     #     for i in range(len(train_x_tensor)):
#     #         train_data.append([train_x_tensor[i], train_y_tensor[i]])
#     #     for i in range(len(test_x_tensor)):
#     #         test_data.append([test_x_tensor[i], test_y_tensor[i]])
#     # #
#     # # # print('adjdg', len(train_data))
#     # #
#     # # train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True)
#     # # test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True)
#     # np.random.shuffle(train_data)
#     # np.random.shuffle(test_data)
#     #
#     # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
#     # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
#
#     all_x_data = x_train + x_test
#     all_y_data = y_train + y_test
#     #
#     x_tensor = torch.FloatTensor(all_x_data)
#     y_tensor = torch.LongTensor(all_y_data)
#     #
#     all_data = []
#     for i in range(len(x_tensor)):
#         all_data.append([x_tensor[i], y_tensor[i]])
#     #
#     np.random.shuffle(all_data)
#     #
#     train_data_subject, _, test_data_subject = all_data[:len(all_data) // 4 * 3], \
#         all_data[len(all_data) // 4 * 3: len(all_data) // 8 * 7], all_data[len(all_data) // 4 * 3:]
#
#     train_loader_subject = DataLoader(dataset=train_data_subject, batch_size=batch_size, shuffle=True)
#     test_loader_subject = DataLoader(dataset=test_data_subject, batch_size=batch_size, shuffle=True)
#     # val_loader_subject = DataLoader(dataset=val_data_subject, batch_size=batch_size, shuffle=True)
#
#     return train_loader_subject, test_loader_subject
#     # return train_loader, test_loader


# def full_HARdata(train_data, test_data, batch):
#     train_data_full = dataProcess(train_data)
#     test_data_full = dataProcess(test_data)
#     train_loader = DataLoader(train_data_full, batch_size=batch, shuffle=False)
#     test_loader = DataLoader(test_data_full, batch_size=batch, shuffle=False)
#
#     return train_loader, test_loader
#
# class FedDataset(Dataset):
#     def __init__(self, dataset, indx, flip):
#         self.dataset = dataset
#         self.indx = [int(i) for i in indx]
#         self.label_flip = flip
#
#     def __len__(self):
#         return len(self.indx)
#
#     def __getitem__(self, item):
#         images, label = self.dataset[self.indx[item]]
#         if self.label_flip == True:
#             label = 7 - label
#         # print(label)
#         return torch.tensor(images).clone().detach(), torch.tensor(label).clone().detach()
#
#
# def get_data(dataset, indeces, batch_size, flip_signal):
#     return DataLoader(FedDataset(dataset, indeces, flip=flip_signal), batch_size=batch_size, shuffle=True)


# def readData():
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


def readData():

    features = pd.read_csv('./UCI HAR Dataset/features.txt', sep='\s+', index_col=0, header=None)
    # train_data = pd.read_csv('./UCI HAR Dataset/train/X_train.txt', sep='\s+',
    #                          names=list(features.values.ravel()))
    train_data = pd.read_csv('UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
    # test_data = pd.read_csv('./UCI HAR Dataset/test/X_test.txt', sep='\s+',
    #                         names=list(features.values.ravel()))
    test_data = pd.read_csv('UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)

    train_label = pd.read_csv('./UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None)
    test_label = pd.read_csv('./UCI HAR Dataset/test/y_test.txt', sep='\s+', header=None)

    train_subject = pd.read_csv('./UCI HAR Dataset/train/subject_train.txt', sep='\s+', header=None)
    test_subject = pd.read_csv('./UCI HAR Dataset/test/subject_test.txt', sep='\s+', header=None)

    label_name = pd.read_csv('UCI HAR Dataset/activity_labels.txt', sep='\s+', header=None, index_col=0)

    train_data['label'] = train_label
    test_data['label'] = test_label

    train_data['subject'] = train_subject
    test_data['subject'] = test_subject

    def get_label_name(num):
        return label_name.iloc[num - 1, 0]

    train_data['label_name'] = train_data['label'].map(get_label_name)
    test_data['label_name'] = test_data['label'].map(get_label_name)

    # 原来标签为1-6，而算法需要0-5
    train_data['label'] = train_data['label'] - 1
    test_data['label'] = test_data['label'] - 1

    np.random.shuffle(train_data.values)
    np.random.shuffle(test_data.values)

    return train_data, test_data


def dataProcess(data):
    x_data = [d[:-3] for d in data.values]
    y_data = [d[-3] for d in data.values]

    x_tensor = torch.FloatTensor(x_data)
    y_tensor = torch.LongTensor(y_data)

    data_tensor = []
    for i in range(len(x_tensor)):
        data_tensor.append([x_tensor[i], y_tensor[i]])
    np.random.shuffle(data_tensor)

    return data_tensor


def generateData(train_data, test_data, subject, batch_size, flip_signal):
    x_train = [d[:-3] for d in train_data.values if d[-2] == subject]
    y_train = [d[-3] for d in train_data.values if d[-2] == subject]
    x_test = [d[:-3] for d in test_data.values if d[-2] == subject]
    y_test = [d[-3] for d in test_data.values if d[-2] == subject]

    all_x_data = x_train + x_test
    all_y_data = y_train + y_test

    if flip_signal:
        for i in range(len(all_y_data)):
            all_y_data[i] = 5 - all_y_data[i]

    x_tensor = torch.FloatTensor(all_x_data)
    y_tensor = torch.LongTensor(all_y_data)

    all_data = []
    for i in range(len(x_tensor)):
        all_data.append([x_tensor[i], y_tensor[i]])

    np.random.shuffle(all_data)

    train_data_subject, val_data_subject, test_data_subject = all_data[:len(all_data) // 4 * 3], \
                                                              all_data[len(all_data) // 4 * 3: len(all_data) // 8 * 7], all_data[len(all_data) // 4 * 3:]

    train_loader_subject = DataLoader(dataset=train_data_subject, batch_size=batch_size, shuffle=True)
    test_loader_subject = DataLoader(dataset=test_data_subject, batch_size=batch_size, shuffle=True)
    val_loader_subject = DataLoader(dataset=val_data_subject, batch_size=batch_size, shuffle=True)

    return train_loader_subject, val_loader_subject, test_loader_subject
