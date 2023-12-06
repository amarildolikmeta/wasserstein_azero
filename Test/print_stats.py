import argparse
from os import listdir
from os.path import dirname

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import figaspect
from pandas import read_csv, concat


def argument_parser():
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--path",  # name on the CLI - drop the `--` for positional/required parameters
        type=str,
        help="path to dir")

    args = CLI.parse_args()
    args = vars(args)

    return args


def print_moving_averages(df_means, df_stds, n_simulations, y_label_name, path):
    c_blue = '#003f5c'

    plt.rcParams.update({'font.size': 17})

    w, h = figaspect(1 / 2)
    plt.figure(dpi=400, figsize=(w, h))
    plt.title("Average of {} runs".format(n_simulations))
    plt.xlabel("Epoch")
    plt.ylabel(y_label_name)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.rc('grid', linestyle="-", color='gray')
    # plt.xscale('log')

    if y_label_name == "solved": plt.ylim([0, 1.1])

    running_mean = df_means[y_label_name]
    running_STD = df_stds[y_label_name]

    len_running_mean = len(running_mean)

    plt.plot(np.arange(len_running_mean), running_mean, color=c_blue)
    plt.fill_between(np.arange(len_running_mean), running_mean - running_STD,
                     running_mean + running_STD, color=c_blue, alpha=0.2)

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

    monitors = []

    for filename in listdir(path):
        if filename.endswith(".csv"):
            content = read_csv(path + "/" + filename)

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

    N = len(monitors)

    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    df_CI = 1.96 * by_row_index.std().replace(np.nan, 0) / np.sqrt(N)

    name = dirname(path)
    name = name.split("/")[-1]

    df_means.to_csv("{}{}_{}_union_means.ccsv".format(args["path"], N, name), index=False)
    df_means.to_csv("{}{}_{}_union_means.ccsv".format("toMove/", N, name), index=False)
    df_CI.to_csv("{}{}_{}_union_CI.ccsv".format("toMove/", N, name), index=False)
    df_CI.to_csv("{}{}_{}_union_CI.ccsv".format(args["path"], N, name), index=False)

    print_moving_averages(df_means, df_CI, N, "solved", path)
    print_moving_averages(df_means, df_CI, N, "length", path)
    print_moving_averages(df_means, df_CI, N, "return", path)

    print_moving_averages_time(df_means, df_CI, N, "solved", path)
    print_moving_averages_time(df_means, df_CI, N, "length", path)
    print_moving_averages_time(df_means, df_CI, N, "return", path)


if __name__ == '__main__': main()
