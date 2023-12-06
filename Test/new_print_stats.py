from os import listdir
from os.path import dirname

import matplotlib.pyplot as plt
import numpy as np
import argparse

from matplotlib.figure import figaspect
from pandas import read_csv, Series, DataFrame, concat

from collections import defaultdict

def argument_parser():
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--path",  # name on the CLI - drop the `--` for positional/required parameters
        type=str,
        help="path to dir")

    CLI.add_argument(
        "--key",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        help="name of variable parameter")

    CLI.add_argument(
        "--n_key",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        help="position of variable parameter")
    CLI.add_argument(
        "--smooth",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        default=0,
        help="running averages")

    args = CLI.parse_args()
    args = vars(args)

    return args

def printdio(dataMean, dataCI, path, N):
    plt.rcParams.update({'font.size': 17})
    w, h = figaspect(1 / 2)

    plt.figure(dpi=400, figsize=(w, h))
    #plt.title("Average of {} runs".format(int(N)))
    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.rc('grid', linestyle="-", color='gray')

    colors = defaultdict()

    for key in dataMean.keys():

        running_mean = dataMean[key][0]["return"]
        if N != 0: running_mean = np.convolve(running_mean, np.ones(N) / N, mode='valid')
        running_variance = dataCI[key][0]["return"]
        if N != 0: running_variance = np.convolve(running_variance, np.ones(N) / N, mode='valid')

        linestyle = "solid"

        if not key in colors:
            p = plt.plot(range(len(running_mean)), running_mean, label=key, linestyle=linestyle)
            colors[key] = p[0].get_color()
        else:
            p = plt.plot(range(len(running_mean)), running_mean, label=key, linestyle=linestyle,
                         color=colors[key])

        plt.fill_between(range(len(running_mean)), running_mean - running_variance,
                         running_mean + running_variance, alpha=0.2, facecolor=p[0].get_color())

    plt.grid(True)
    plt.legend()

    plt.savefig(path + "/return.pdf", bbox_inches='tight')
    plt.close("all")

    plt.figure(dpi=400, figsize=(w, h))
    #plt.title("Average of {} runs".format(int(N)))
    plt.xlabel("Epoch")
    plt.ylabel("Solved Episodes (%)")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.rc('grid', linestyle="-", color='gray')

    for key in dataMean.keys():

        running_mean = dataMean[key][0]["solved"]*100
        if N != 0: running_mean = np.convolve(running_mean, np.ones(N) / N, mode='valid')
        running_variance = dataCI[key][0]["return"]
        if N != 0: running_variance = np.convolve(running_variance, np.ones(N) / N, mode='valid')

        linestyle = "solid"

        if not key in colors:
            p = plt.plot(range(len(running_mean)), running_mean, label=key, linestyle=linestyle)
            colors[key] = p[0].get_color()
        else:
            p = plt.plot(range(len(running_mean)), running_mean, label=key, linestyle=linestyle,
                         color=colors[key])

        plt.fill_between(range(len(running_mean)), running_mean - running_variance,
                         running_mean + running_variance, alpha=0.2, facecolor=p[0].get_color())

    plt.grid(True)

    plt.savefig(path + "/solved.pdf", bbox_inches='tight')
    plt.close("all")

def main():

    args = argument_parser()
    path = args["path"]

    dataMean = defaultdict(list)
    dataCI = defaultdict(list)

    for filename in listdir(path):
        if filename.endswith(".ccsv"):
            content = read_csv(path+"/"+filename)

            # search_bit_epoch_HER_K
            filename = filename.split(".ccsv")[:-1]
            filename = "_".join(filename)
            params = filename.split("_")
            new_key = params[args["n_key"]]

            if params[-1] == "CI":
                dataCI[new_key].append(content)
            else:
                dataMean[new_key].append(content)

    printdio(dataMean, dataCI, path, args["smooth"])

    #printQC(dataCI, path)
    #printNVP(dataMean, dataCI, path)
    #printPosterior(dataMean, dataCI, path)

if __name__ == '__main__': main()
