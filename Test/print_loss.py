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

def print_moving_averages(cont, y_label_name, path):
    c_blue = '#003f5c'

    plt.rcParams.update({'font.size': 17})

    w, h = figaspect(1 / 2)
    plt.figure(dpi=400, figsize=(w, h))
    plt.xlabel("Epoch")
    plt.ylabel(y_label_name)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.rc('grid', linestyle="-", color='gray')
    # plt.xscale('log')

    running_mean = cont[y_label_name]

    len_running_mean = len(running_mean)

    plt.plot(np.arange(len_running_mean), running_mean, color=c_blue)

    plt.grid(True)

    plt.savefig(path + "{}.pdf".format(y_label_name), bbox_inches='tight')

    plt.close("all")


def main():

    args = argument_parser()
    path = args["path"]

    content = read_csv(path, sep=',', skipinitialspace=True, decimal='.')

    for column in content:
        print_moving_averages(content, column, "")

    #print_moving_averages(content, "loss", "")
    #print_moving_averages(content, "value_loss", "")
    #print_moving_averages(content, "pol_loss", "")


    return 0

if __name__ == '__main__': main()