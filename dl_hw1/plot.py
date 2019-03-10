from matplotlib import pyplot as plt
import json

n = 50000
def plot(title, fig_title):
    json_path = './res_json/' + title + '.json'
    fig_savepath = './' + title.split('_')[0] + '/' + title + '.png'
    with open(json_path) as f:
        loss_curve = json.load(f)
    idx = [loss_curve[i][1]/n for i in range(len(loss_curve))]
    loss = [loss_curve[i][2] for i in range(len(loss_curve))]

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim([0, 3])
    plt.title(fig_title)
    plt.plot(idx, loss)
    plt.savefig(fig_savepath)
    plt.close()

if __name__ == '__main__':
    res = ['baseline'] + ['q'+str(i)+'_'+str(j) for i in range(1, 6) for j in range(1, 3)]
    fig_titles = ['baseline', 'add one hidden layer', 'drop one hidden layer', 'BGD', 'SGD',
                  'uniform initialization', 'truncated_normal initiallization', 'Momentum', 'Adam',
                  'l2 regularization', 'dropout']

    for i in range(len(res)):
        plot(res[i], fig_titles[i])