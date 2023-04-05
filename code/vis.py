import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

unormal = [16, 51, 56]
station_array = [0, 1, 2, 3, 11, 16, 17, 18, 23, 24, 26, 27,
                 28, 30, 34, 35, 37, 39, 41, 42, 45, 46, 49, 52, 54]

model_dict = {
    'VanillaRNN': 0,
    'LSTM': 0,
    'BiLSTM': 0,
}

def rename():
    import shutil
    # trained_list = [17, 18, 24, 30, 35, 37, 51]
    for iidx in range(32, 31, -1):
        for file in os.listdir('./stat/'):
            idx = int(file.split('_')[3])
            if idx == iidx:
                shutil.copy(os.path.join('./stat/', file), os.path.join('./stat/', file.replace(str(idx), str(idx+1), 1)))
        for file in os.listdir('./checkpoints/'):
            idx = int(file.split('_')[2])
            if idx == iidx:
                shutil.copy(os.path.join('./checkpoints/', file), os.path.join('./checkpoints/', file.replace(str(idx), str(idx+1), 1)))
        for file in os.listdir('./vis/'):
            idx = int(file.split('_')[2])
            if idx == iidx:
                shutil.copy(os.path.join('./vis/', file), os.path.join('./vis/', file.replace(str(idx), str(idx+1), 1)))


def stat_merge(model):
    stat_df = None
    stat_mean = None
    for idx in station_array:
        for cv in range(5):
            df = pd.read_csv('./stat/stat_{}_idx_{}_cv_{}.csv'.format(model, idx, cv))
            if stat_df is None:
                stat_df = df
            else:
                stat_df = stat_df.append(df)

        if stat_mean is None:
            stat_mean = pd.DataFrame(data=np.reshape(stat_df.mean().values, (1, -1)), columns=list(stat_df.columns))
        else:
            stat_mean = stat_mean.append(pd.DataFrame(data=np.reshape(stat_df.iloc[-5:].mean().values, (1, -1)), columns=list(stat_df.columns)))

    stat_df.to_csv('{}_stat.csv'.format(model), index=False)
    stat_mean.to_csv('{}_stat_mean.csv'.format(model), index=False)
    stat_mean.to_csv('{}_stat_var.csv'.format(model), index=False)

def EMA(curve):
    alpha = 0.2
    state = 0
    smoothed_curve = []
    for item in curve:
        if len(smoothed_curve) == 0:
            state = item
        else:
            state = alpha * item + state * (1 - alpha)
        smoothed_curve.append(state)
    return np.array(smoothed_curve)

def revis():
    for model in model_dict.keys():
        for idx in station_array:
            for cv in range(5):
                df = pd.read_csv('./stat/history_{}_idx_{}_cv_{}.csv'.format(model, idx, cv))
                plt.figure(1)
                plt.plot(EMA(df['train acc'].values))
                plt.plot(EMA(df['valid acc'].values))
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['train acc', 'valid acc'])
                plt.savefig('vis/Accuracy_{}_{}_{}.png'.format(model, idx, cv))
                plt.close()

                plt.figure(2)
                plt.plot(EMA(df['train loss'].values))
                plt.plot(EMA(df['valid loss'].values))
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['train loss', 'valid loss'])
                plt.savefig('vis/loss_{}_{}_{}.png'.format(model, idx, cv))
                plt.close()

def draw():
    for idx in station_array:
        for cv in range(5):
            for metrics in [['train loss', 'train acc'], ['valid loss', 'valid acc']]:
                plt.figure()
                plt.xlabel('Epoch')
                flag = True
                for metric in metrics:
                    for model in ['VanillaRNN', 'LSTM', 'BiLSTM']:
                        df = pd.read_csv('stat/history_{}_idx_{}_cv_{}.csv'.format(model, idx, cv))
                        plt.plot(EMA(df[metric].values.flatten()))
                        plt.ylabel(metric)
                    if flag:
                        plt.twinx()
                        flag = False
                plt.grid()
                plt.legend(['VanillaRNN', 'LSTM', 'BiLSTM'])
                plt.savefig('sample {} {}.png'.format(idx, cv), dpi=800)
                plt.show()
                plt.close()


def vis():
    array = np.array([-0.2, 0, 0.2])
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    models = ['VanillaRNN', 'LSTM', 'BiLSTM']
    for metric in metrics:
        plt.figure(figsize=(20, 8))
        for idx, model in enumerate(models):
            df = pd.read_csv('{}_stat_mean.csv'.format(model))
            plt.bar(np.arange(len(station_array)) + array[idx], df[metric], width=0.2)

        plt.xticks(np.arange(len(station_array)), station_array)
        plt.xlabel('Station Index')
        plt.legend(models, loc='upper right')
        plt.title(metric)
        plt.grid()
        plt.savefig('all stations {}.png'.format(metric))
        plt.show()


    array = np.array([-0.3, -0.1, 0.1, 0.3])
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        stat_df = None
        for idx in station_array:
            for cv in range(5):
                df = pd.read_csv('./stat/stat_{}_idx_{}_cv_{}.csv'.format(model, idx, cv))
                if stat_df is None:
                    stat_df = df
                else:
                    stat_df = stat_df.append(df)

        for j, metric in enumerate(metrics):
            print(len(stat_df[metric].values))
            ax.boxplot(stat_df[metric], widths=0.15, positions=[i + array[j]], medianprops={'color': "C{}".format(j)}, \
                        boxprops=dict(color="C{}".format(j)), capprops={'color': "C{}".format(j)}, \
                        flierprops={'color': "C{}".format(j)}, whiskerprops={'color': "C{}".format(j)})

    plt.grid()
    plt.xticks(np.arange(3), ['Vanilla RNN', 'LSTM', 'Bi-LSTM'])
    plt.legend(metrics, loc='upper right')
    plt.savefig('models.png')
    plt.show()

    array = np.array([-0.3, 0, 0.3])
    for metric in metrics:
        plt.figure(figsize=(20, 8))
        for i, model in enumerate(models):
            stat_idx = []
            for idx in station_array:
                stat_df = None
                for cv in range(5):
                    df = pd.read_csv('./stat/stat_{}_idx_{}_cv_{}.csv'.format(model, idx, cv))
                    if stat_df is None:
                        stat_df = df
                    else:
                        stat_df = stat_df.append(df)

                stat_idx.append(stat_df[metric].values)

            plt.boxplot(stat_idx, widths=0.2, positions=np.arange(len(station_array)) + array[i], medianprops={'color': "C{}".format(i)}, \
                        boxprops=dict(color="C{}".format(i)), capprops={'color': "C{}".format(i)}, \
                        flierprops={'color': "C{}".format(i)}, whiskerprops={'color': "C{}".format(i)})

        plt.xticks(np.arange(len(station_array)), station_array)
        plt.grid()
        plt.xlabel('Station Index')
        plt.legend(models, loc='upper right')
        plt.title(metric)
        plt.savefig('boxplot {}.png'.format(metric))
        plt.show()


def comp():
    for model in model_dict.keys():
        pd.read_csv('{}_stat_mean.csv'.format(model)).mean().to_csv('{}.csv'.format(model))

if __name__ == '__main__':
    # for model in model_dict.keys():
    #     stat_merge(model)
    # vis()
    draw()