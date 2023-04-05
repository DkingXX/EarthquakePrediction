import torch
import os
import logging
import nni
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools import EarlyStopping
from Models import VanillaRNN, LSTMModel, BiLSTMModel
from preprocess import get_dataloader, load_data
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import argparse
# hyper params
model_dict = {
    'VanillaRNN': VanillaRNN,
    'LSTM': LSTMModel,
    'BiLSTM': BiLSTMModel,
}

unormal = [16, 51, 56]
station_array = [0, 1, 2, 3, 11, 16, 17, 18, 23, 24, 26, 27,
                 28, 30, 34, 35, 37, 39, 41, 42, 45, 46, 49, 52, 54]
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--decay', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--idx', type=int, default=51)
parser.add_argument('--model', type=str, default='VanillaRNN')
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=0)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--opt', type=bool, default=False)


LOG = logging.getLogger('example')

optimizers={'SGD': torch.optim.SGD,
            'RMSProp': torch.optim.RMSprop,
            'Adam': torch.optim.Adam}

# 训练过程
def training(args, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 确认执行设备
    model = model_dict[args.model](params['dropout_rate']).to(device)
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(idx=args.idx, cv=args.cv, test=False) # 得到数据的迭代器，这里训练用train_dataloader
    # 验证用valid_dataloader 测试用test_dataloader
    loss_func = torch.nn.BCELoss().to(device) # loss计算函数
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay']) # 模型参数优化器
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer) # 学习率调整器
    early_stopping = EarlyStopping(patience=args.patience)
    best_loss, best_acc = 100000, 0 # 用于判断验证集loss最小的参数
    history = {'train loss': [], 'valid loss': [], 'train acc': [], 'valid acc': []} # 保存历史loss
    if not os.path.exists('./checkpoints'): # 创建模型保存文件的文件夹
        os.mkdir('./checkpoints')
    for epoch in tqdm(range(args.epochs)):
        mean_train_loss = [] # 保存这个epoch中的训练集loss，用于绘图
        mean_test_loss = []# 保存历史的验证集loss，用于绘图

        labels = []
        preds = []
        model.train() # model切换为train模式
        for item in train_dataloader: # 遍历数据集，获取Sample
            optimizer.zero_grad() # 清除优化器的梯度，这是每次更新参数必需的操作
            value, label = item[0], item[1] # 获取数据集sample中的value和label
            output = model(value.to(device)) # 得到输出

            loss = loss_func(output.squeeze(), label.to(device)) # 由 output和label计算loss
            loss.backward() # 由loss进行BP得到梯度
            optimizer.step() # 优化器更新参数
            mean_train_loss.append(loss.detach().cpu().numpy()) # 把loss放入历史信息
            output = output.cpu().detach().numpy()
            output[output < 0.5] = 0
            output[output >= 0.5] = 1
            preds.extend(output)
            labels.extend(label.numpy()) # 保存label，以便计算当前轮的准确率

        test_preds = []
        test_labels = []
        model.eval() # model切换为评估模式
        for item in valid_dataloader: # 遍历验证集
            value, label = item[0], item[1] # 获取数据
            output = model(value.to(device)) # 得到输出
            loss = loss_func(output.squeeze(), label.to(device)) # 计算loss
            mean_test_loss.append(loss.detach().cpu().numpy()) # 保存loss
            output = output.cpu().detach().numpy()
            output[output < 0.5] = 0
            output[output >= 0.5] = 1
            test_preds.extend(output)
            test_labels.extend(label.numpy())
        # 上面的mean_train_loss和mean_test_loss是保存一个epoch内的loss信息，历史信息保存的是每个epoch的loss
        history['train loss'].append(np.mean(mean_train_loss))
        history['valid loss'].append(np.mean(mean_test_loss))
        history['train acc'].append(accuracy_score(labels, preds))
        history['valid acc'].append(accuracy_score(test_labels, test_preds))
        nni.report_intermediate_result(history['valid acc'][-1])

        if args.opt == False:
            print(  # 打印该epoch的两个loss和两个accuracy
            'Epoch {}/{}: train loss is {:.4f}, valid loss is {:.4f}, train acc is {:.4f}, valid acc is {:.4f}'
                .format(epoch + 1, 100, np.mean(mean_train_loss), np.mean(mean_test_loss),
                        accuracy_score(labels, preds), accuracy_score(test_labels, test_preds)))
        # lr_scheduler.step()  # 更新学习率
        if np.mean(mean_test_loss) < best_loss: # 这是判断保存验证集loss最小模型的操作
            best_loss = np.mean(mean_test_loss)
            best_acc = accuracy_score(test_labels, test_preds)

            torch.save(model, os.path.join('./checkpoints/',
                                           '{}_idx_{}_cv_{}'.format(args.model, args.idx, args.cv)))

        early_stopping(np.mean(mean_test_loss), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if args.opt == False:
        # 画出历史两个loss的折线图并保存下来
        plt.figure()
        plt.plot(np.arange(0, len(history['train loss'])), history['train loss']) # 折线图绘制
        plt.plot(np.arange(0, len(history['valid loss'])), history['valid loss'])
        plt.legend(['train loss', 'valid loss'])
        plt.title('Loss') # 标题
        plt.savefig('./vis/loss_{}_{}_{}.png'.format(args.model, args.idx, args.cv)) # 保存
        plt.close()

        # 画出历史两个acc的折线图并保存下来
        plt.figure()
        plt.plot(np.arange(0, len(history['train acc'])), history['train acc']) # 折线图绘制
        plt.plot(np.arange(0, len(history['valid acc'])), history['valid acc'])
        plt.legend(['train acc', 'valid acc'])
        plt.title('Accuracy') # 标题
        plt.savefig('./vis/Accuracy_{}_{}_{}.png'.format(args.model, args.idx, args.cv)) # 保存
        plt.close()

    ############ testing ############
    if args.opt == False:
        pd.DataFrame(history).to_csv(os.path.join('./stat/', 'history_{}_idx_{}_cv_{}.csv'.format(
            args.model, args.idx, args.cv)), index=False)

    test_preds = []
    test_labels = []
    model = torch.load(os.path.join('./checkpoints/',
                                               '{}_idx_{}_cv_{}'.format(args.model, args.idx, args.cv))).to(device)
    model.eval()  # model切换为评估模式
    for item in test_dataloader:  # 遍历验证集
        value, label = item[0], item[1]  # 获取数据
        output = model(value.to(device))  # 得到输出
        output = output.cpu().detach().numpy()
        output[output < 0.5] = 0
        output[output >= 0.5] = 1
        test_preds.extend(output)
        test_labels.extend(label.numpy())
    cp = classification_report(test_labels, test_preds, digits=4, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).flatten()
    dic = {
        'accuracy': cp['accuracy'],
        'precision': cp['weighted avg']['precision'],
        'recall': cp['weighted avg']['recall'],
        'f1-score': cp['weighted avg']['f1-score'],
        'FN': tn,
        'FP': fp,
        'TN': fn,
        'TP': tp,
    }
    if args.opt == False:
        pd.DataFrame(dic, index=[0]).to_csv(os.path.join('./stat/', 'stat_{}_idx_{}_cv_{}.csv'.format(
            args.model, args.idx, args.cv)), index=False)

    return best_acc

def cv_test(args, params):
    for idx in station_array:
        args.idx = idx
        print(idx, args.model)
        for cv in range(0, 5):
            args.cv = cv
            training(args, params)

def generate_default_params():
    params = {
        'dropout_rate': 0.2,
        'learning_rate': 1e-4,
        'weight_decay': 0.001,
    }
    return params
        
def hypertune(args):

    # get parameters from tuner
    RECEIVED_PARAMS = nni.get_next_parameter()
    LOG.debug(RECEIVED_PARAMS)
    PARAMS = generate_default_params()
    PARAMS.update(RECEIVED_PARAMS)
    # train
    accs = []
    for args.cv in range(5):
        accs.append(training(args, PARAMS))
    nni.report_final_result(np.mean(accs))

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 确认执行设备
    model = torch.load(os.path.join('./checkpoints/',
                                               '{}_idx_{}_cv_{}'.format(args.model, args.idx, args.cv))).to(device)
    print(args.idx)
    test_dataloader = get_dataloader(idx=args.idx, cv=args.cv, test=True) # 仅获得测试数据的dataloader
    test_preds = []
    test_labels = []
    model.eval()  # model切换为评估模式
    for item in test_dataloader:  # 遍历验证集
        value, label = item[0], item[1]  # 获取数据
        output = model(value.to(device))  # 得到输出
        output = output.cpu().detach().numpy()
        output[output < 0.5] = 0
        output[output >= 0.5] = 1
        test_preds.extend(output)
        test_labels.extend(label.numpy())
    cp = classification_report(test_labels, test_preds, digits=4, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).flatten()
    dic = {
        'accuracy': cp['accuracy'],
        'precision': cp['weighted avg']['precision'],
        'recall': cp['weighted avg']['recall'],
        'f1-score': cp['weighted avg']['f1-score'],
        'FN': tn,
        'FP': fp,
        'TN': fn,
        'TP': tp,
    }
    if args.opt == False:
        pd.DataFrame(dic, index=[0]).to_csv(os.path.join('./stat/', 'stat_{}_idx_{}_cv_{}.csv'.format(
            args.model, args.idx, args.cv)), index=False)

def test_all():
    idx_list = np.arange(57)
    idx_list = np.delete(idx_list, 32)
    idx_list = np.delete(idx_list, 12)
    for args.idx in idx_list:
        for args.cv in range(5):
            test(args)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.opt == True:
        hypertune(args)
    else:
        params = generate_default_params()
        cv_test(args, params)
    # cv_test(args, params=generate_default_params())
    # revis()
    # rename()