from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.figure import figaspect
from pandas import read_csv, concat


def _running_mean(x, N):
    divider = np.convolve(np.ones_like(x), np.ones((N,)), mode='same')
    return np.convolve(x, np.ones((N,)), mode='same') / divider


def argument_parser():
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--path",  # name on the CLI - drop the `--` for positional/required parameters
        type=str,
        help="path to dir")
    CLI.add_argument(
        "--N",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        default='1',
        help="running window width")
    CLI.add_argument(
        "--freq",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        default='1',
        help="freq of evaluation")

    args = CLI.parse_args()
    args = vars(args)

    return args

name_to_label = {
    "solved": "Solved Episodes (%)",
    "return": "Average Return"
}
def print_moving_averages(list_of_means, list_of_stds, n_simulations, y_label_name, path, N=5, freq=1, labels=[],
                          min_len=1000):
    c_blue = '#003f5c'
    plt.rcParams.update({'font.size': 17})
    if y_label_name == 'solved':
        factor = 100
    else:
        factor = 1
    w, h = figaspect(1 / 2)
    plt.figure(dpi=400, figsize=(w, h))
    # plt.title("Average of {} runs".format(n_simulations))
    plt.xlabel("Epoch")
    plt.ylabel(name_to_label[y_label_name])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.rc('grid', linestyle="-", color='gray')
    for i in range(len(list_of_means)):
        df_means = list_of_means[i]
        df_stds = list_of_stds[i]
        data = df_means[y_label_name] * factor
        data_std = df_stds[y_label_name] * factor
        if freq > 1:
            data = data[::freq]
            data_std = data_std[::freq]
        #min_len = 1500
        running_mean = _running_mean(data, N)[:min_len]
        running_variance = _running_mean(data_std, N)[:min_len]
        len_running_mean = len(running_mean)
        xs = np.arange(len_running_mean) * freq
        plt.plot(xs, running_mean, label=labels[i])
        plt.fill_between(xs, running_mean - 2 * running_variance / np.sqrt(n_simulations),
                         running_mean + 2 * running_variance / np.sqrt(n_simulations), alpha=0.2)

    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig(path + "/{}.pdf".format(y_label_name), bbox_inches='tight')

    plt.close("all")


def main():

    args = argument_parser()
    path_ = args["path"]
    window_width = args['N']
    settings = ['iclr', 'future_noisy', 'future_P', 'future_all_P']
    labels = ['ICLR', 'Future-Noisy', 'Future-P', 'Future-All-P']
    # if "Point" in path_:
    #     settings = ['dqn_her_baselines', 'dqn_her_baselines_our_net', 'dqn_her_paper', 'dqn_her_paper_our_net']
    #     labels = ['dqn_her_baselines', 'dqn_her_baselines_our_net', 'dqn_her_paper', 'dqn_her_paper_our_net']
    # elif "Maze" in path_:
    #     settings = ['dqn_her_baselines', 'dqn_her_paper']
    #     labels = ['dqn_her_baselines',  'dqn_her_paper']
    settings = ['best_point']  # 'best_param',, 'best_bitflip']
    labels = ['DQN+HER']  # , 'QC_2', 'bitflip-params']
    # settings = ['50', '60', '70']
    # labels = ['50 bits', '60 bits', '70 bits']
    list_of_means = []
    list_of_stds = []
    min_len = 10000
    for setting in settings:
        monitors = []
        path = path_ + "/" + setting
        found = False
        try:
            for filename in listdir(path):
                found = True
                if filename.endswith(".csv"):
                    content = read_csv(path+"/"+filename)
                    # Evaluate time
                    column_names = ["t-expanded", "t-training", "t-pitting", "t-testing"]
                    content['tot_time'] = 0
                    for name in column_names:
                        if name in content: content['tot_time'] += content[name]
                    content["tot_time"] = content["tot_time"].cumsum()
                    content = content[["return", "solved", "length", "tot_time"]]
                    monitors.append(content)
        except:
            found = False
        # concatenate them
        if found:
            df_concat = concat(monitors)
            by_row_index = df_concat.groupby(df_concat.index)
            df_means = by_row_index.mean()
            df_stds = by_row_index.std().replace(np.nan, 0)
            N = len(monitors)
            list_of_means.append(df_means)
            list_of_stds.append(df_stds)
            if df_means.shape[0] < min_len:
                min_len = df_means.shape[0]
    min_len = min_len // args['freq'] + 1
    print_moving_averages(list_of_means, list_of_stds, N, "solved", path_, N=window_width, freq=args['freq'],
                          labels=labels, min_len=min_len)
    print_moving_averages(list_of_means, list_of_stds, N, "return", path_, N=window_width, freq=args['freq'],
                          labels=labels, min_len=min_len)
    # print_moving_averages_time(df_means, df_stds, N, "solved", path)
    # print_moving_averages_time(df_means, df_stds, N, "length", path)
    # print_moving_averages_time(df_means, df_stds, N, "return", path)

if __name__ == '__main__': main()