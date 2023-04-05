import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import os
import random
import pickle

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(idx):
    X, Y = [], []
    data = pickle.load(open('./data/sc_seismic_normal.pkl', 'rb'))
    for item in data:
        X.append(item)
    data = X[idx]
    X = []
    for item in data:
        X.append(item[0])

    X = np.expand_dims(np.array(X), 1)

    X1 = []
    data1 = np.load('./data/sc_seismic_earthquake.pkl', allow_pickle=True)
    for item in data1:
        X1.append(item)
    data = X1[idx]
    X1 = []
    for item in data:
        X1.append(item[0])

    X1 = np.array(X1)

    Y.extend(np.zeros(len(X1))) # 地震数据对应的label为0

    X = np.concatenate([X[:len(X1)], np.expand_dims(X1, 1)], axis=0)
    Y.extend(np.ones(len(X) - len(Y))) # 正常数据对应的label为1
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    # X = (X - np.min(X)) / (np.max(X) - np.min(X))
    # 因为X是一整块正常的数据和一整块地震数据，这里做shuffle，把X和Y以相同的顺序打乱
    shuffle_ix = np.random.permutation(np.arange(len(X)))
    data = X[shuffle_ix]
    label = Y[shuffle_ix]
    return data, label

# 生成dataloader，dataloader是torch框架中数据管道的类，用于快速遍历数据
def get_dataloader(batch_size=32, idx=0, cv=0, test=False):
    # 固定随机种子，保证每次数据分割是一致的
    seed_torch()

    x, y = load_data(idx)
    x_train_valid, y_train_valid = x[:int(0.9 * len(x))], y[:int(0.9 * len(x))]
    x_test, y_test = x[int(0.9 * len(x)):], y[int(0.9 * len(x)):]
    length = len(x_train_valid) // 5

    x_valid = x_train_valid[cv*length: min((cv+1) * length, len(x_train_valid))]
    y_valid = y_train_valid[cv*length: min((cv+1) * length, len(y_train_valid))]
    x_train = np.delete(x_train_valid, np.arange(cv*length, min((cv+1) * length, len(x_train_valid))), axis=0)
    y_train = np.delete(y_train_valid, np.arange(cv*length, min((cv+1) * length, len(y_train_valid))))

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))# 得到数据集
    valid_dataset = TensorDataset(torch.tensor(x_valid), torch.tensor(y_valid))# 得到数据集
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))# 得到数据集

    if test: # 测试的话，意味着有训练好的模型，只需要在测试数据上测试一下即可
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=16, shuffle=False) # 制作
        return test_dataloader
    else: # 训练的话，需要训练计和验证集
        # 转换成dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=16, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=16, shuffle=False)
        return train_dataloader, valid_dataloader, test_dataloader

def sample_vis():
    import matplotlib.pyplot as plt
    station_array = [16, 51, 56]
    X, Y = [], []
    data = pickle.load(open('./data/sc_seismic_normal.pkl', 'rb'))
    print(len(data))
    for idx, item in enumerate(data):
        if idx in station_array:
            X.append(len(item))

    X1 = []
    data1 = np.load('./data/sc_seismic_earthquake.pkl', allow_pickle=True)
    for idx, item in enumerate(data1):
        if idx in station_array:
            X1.append(len(item))
    width = 0.2
    plt.figure(figsize=(20, 8))
    plt.bar(np.arange(len(X)) - width, X, width=2 * width)
    plt.bar(np.arange(len(X1)) + width, X1, width=2 * width)
    plt.xticks(np.arange(len(X)))
    plt.grid()
    plt.xlabel('Station Index')
    plt.ylabel('Number of Sequences')
    plt.legend(['number of normal', 'number of earthquake'])
    plt.xticks(np.arange(len(station_array)), station_array)
    plt.show()

if __name__ == '__main__':
    sample_vis()
    # test_dataloader = get_dataloader(idx=56, cv=0, test=True)
    # for item in test_dataloader:
    #     print(item[0].shape)