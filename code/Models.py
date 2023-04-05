import torch.nn as nn
import torch

length = 1501
class VanillaRNN(nn.Module):
    def __init__(self, dropout_rate=0.2, hidden=128):
        super(VanillaRNN, self).__init__()
        # 每个网络层后面跟的都是数据的shape
        self.rnn1 = nn.RNN(input_size=1, hidden_size=hidden, batch_first=True) # batchsize, length, H_in
        self.dense = nn.Linear(length * hidden, 1)  # batchsize, length * H_out
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid() # batchsize, 1

    def forward(self, x):
        # 原始数据读进来是[data samples, H_in, length]，需要转换维度到[data samples, length, H_in]
        x = x.permute(0, 2, 1)
        x, _ = self.rnn1(x)
        x = torch.flatten(x, 1, -1)
        x = self.dense(self.dropout(x))
        return x

class LSTMModel(nn.Module):
    def __init__(self, dropout_rate=0.2, hidden=128):
        super(LSTMModel, self).__init__()

        self.rnn1 = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True) # batchsize, length, H_in
        self.pooling = nn.AdaptiveAvgPool1d(1) # batchsize, length
        self.dense1 = nn.Linear(length * hidden, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid() # batchsize, 1

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.rnn1(x)
        x = self.dense1(self.dropout(torch.flatten(x, -2, -1)))
        x = self.sigmoid(x)
        return x

class BiLSTMModel(nn.Module):
    def __init__(self, dropout_rate=0.2, hidden=128):
        super(BiLSTMModel, self).__init__()

        self.rnn1 = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True, bidirectional=True)# batchsize, length, H_in
        self.rnn2 = nn.LSTM(input_size=hidden * 2, hidden_size=hidden, batch_first=True, bidirectional=True)# batchsize, length, H_in
        self.dense = nn.Linear(length * 2 * hidden, 1) # batchsize, 1
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = torch.flatten(x, 1, -1)
        x = self.dense(self.dropout(x))
        x = self.sigmoid(x)
        return x
    

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense = nn.Linear(length, 128)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.dense(torch.flatten(x, 1, -1)))
        x = self.dense1(self.sigmoid(x))
        return x

if __name__ == '__main__':
    model = BiLSTMModel()
    x = torch.zeros([1, 58, 61])
    output = model(x)