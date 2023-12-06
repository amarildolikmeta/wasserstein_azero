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


def print_moving_averages(df_means, df_stds, n_simulations, y_label_name, path, N=5, freq=1):
    c_blue = '#003f5c'

    plt.rcParams.update({'font.size': 17})

    w, h = figaspect(1 / 2)
    plt.figure(dpi=400, figsize=(w, h))
    plt.title("Average of {} runs".format(n_simulations))
    plt.xlabel("Epoch")
    plt.ylabel(y_label_name)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.rc('grid', linestyle="-", color='gray')

    data = df_means[y_label_name]
    data_std = df_stds[y_label_name]
    if freq > 1:
        data = data[::freq]
        data_std = data_std[::freq]
    running_mean = _running_mean(data, N)
    running_variance = _running_mean(data_std, N)

    len_running_mean = len(running_mean)

    xs = np.arange(len_running_mean) * freq
    plt.plot(xs, running_mean, color=c_blue)
    plt.fill_between(xs, running_mean - 2 * running_variance / np.sqrt(n_simulations),
                     running_mean + 2 * running_variance / np.sqrt(n_simulations), color=c_blue, alpha=0.2)

    plt.grid(True)

    plt.savefig(path + "/{}.pdf".format(y_label_name), bbox_inches='tight')

    plt.close("all")


def print_moving_averages_time(df_means, df_stds, n_simulations, y_label_name, path):
    c_blue = '#003f5c'

    plt.rcParams.update({'font.size': 17})

    w, h = figaspect(1 / 2)
    plt.figure(dpi=400, figsize=(w, h))
    plt.title("Average of {} runs".format(n_simulations))
    plt.xlabel("Time (s)")
    plt.ylabel(y_label_name)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.rc('grid', linestyle="-", color='gray')

    running_mean = df_means[y_label_name]
    running_variance = df_stds[y_label_name]

    plt.plot(df_means["tot_time"], running_mean, color=c_blue)
    plt.fill_between(df_means["tot_time"], running_mean - running_variance,
                     running_mean + running_variance, color=c_blue, alpha=0.2)

    plt.grid(True)

    plt.savefig(path + "/time_{}.pdf".format(y_label_name), bbox_inches='tight')

    plt.close("all")

def main():

    args = argument_parser()
    path = args["path"]
    window_width = args['N']
    monitors = []

    for filename in listdir(path):
        if filename.endswith(".csv"):
            content = read_csv(path+"/"+filename)

            # Evaluate time
            column_names = ["t-expanded", "t-training", "t-pitting", "t-testing", "t-azVSmcts", "t-azVSpolicy"]
            content['tot_time'] = 0
            for name in column_names:
                if name in content: content['tot_time'] += content[name]
            content["tot_time"] = content["tot_time"].cumsum()

            content = content[["return", "solved", "length", "tot_time"]]


            monitors.append(content)

    # concatenate them
    df_concat = concat(monitors)

    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    df_stds = by_row_index.std().replace(np.nan, 0)

    N = len(monitors)

    print_moving_averages(df_means, df_stds, N, "solved", path, N=window_width, freq=args['freq'])
    print_moving_averages(df_means, df_stds, N, "length", path, N=window_width, freq=args['freq'])
    print_moving_averages(df_means, df_stds, N, "return", path, N=window_width, freq=args['freq'])

    print_moving_averages_time(df_means, df_stds, N, "solved", path)
    print_moving_averages_time(df_means, df_stds, N, "length", path)
    print_moving_averages_time(df_means, df_stds, N, "return", path)

if __name__ == '__main__': main()