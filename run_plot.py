#!/usr/bin/env python3
import os
import numpy as np
import argparse
import textwrap

from matplotlib import pyplot as plt

from sweat4science.evaluation.sessionset import MF_sessionset as mfs
from sweat4science.workspace.Workspace import Workspace
from sweat4science.s4sconfig import workspace_dir

from s4s_rnn import utils, plotting, evaluation


def get_arguments():
    parser = argparse.ArgumentParser(description=textwrap.dedent('''
        Script to run plotting functions
        '''))
    parser.add_argument('--scenario', '-s', type=str,
                        choices=['hbm_velocity', 'epoch_error'],
                        default='hbm_velocity',
                        help='choice of plotting function')
    parser.add_argument('--file', '-f', type=argparse.FileType('rb'),
                        help="training log file to plot errors by epoch, required for epoch_error scenario")
    return parser


def plot_hbm_velocity(eval_dict, session_list):
    for s_eval in map(eval_dict.get, session_list):
        data = utils.get_data_from_session(s_eval.session)
        plotting.plot_multi_y(data[:, -1], data[:, 1], "Time (s)", "Heart rate (hbm)", "Velocity (m/s)",
                s_eval.session.name, np.arange(0, data.shape[0]*10, 10), y1_range=(80, 200), y2_range=(-1, 5))
        pass
    return


def plot_epoch(log_file):
    train_info = np.genfromtxt(log_file, delimiter=",", skip_header=1)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)

    max_y = np.max(train_info[:, 1:3])
    min_y = np.min(train_info[:, 1:3])
    difference = max_y - min_y
    ax.set_ylim([min_y - difference*0.1, max_y + difference*0.1])

    epochs = list(map(int, train_info[:, 0]))
    acc_line, = ax.plot(epochs, train_info[:, 1], '-r+',
                         markersize=4, label='train accuracy')
    val_line, = ax.plot(epochs, train_info[:, 2], '-g+',
                         markersize=4, label='validation accuracy')
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(handles=[acc_line, val_line], loc='upper center',
              bbox_to_anchor=(0.5, -0.08), fancybox=True,
              shadow=True, ncol=2)
    plt.title('Validation and training accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.grid()
    plt.show()
    return


def main(parser):
    arguments = parser.parse_args()

    if arguments.scenario == 'hbm_velocity':
        workspace_folder = os.path.join(workspace_dir, "session-data")
        ws = Workspace(workspace_folder)
        sessions = mfs.ICT_indoor(ws)
        eval_dict = evaluation.ExperimentEvalutationDict(sessions)
        plot_hbm_velocity(eval_dict, ['session_01_20150219', 'session_03_20150305', 'lactate_test_20150408'])
    elif arguments.scenario == 'epoch_error':
        if arguments.file is None:
            print("epoch_error scenario requires argument log file")
            parser.print_usage()
            return
        plot_epoch(arguments.file)
        pass
    return


if __name__ == "__main__":
    parser = get_arguments()
    main(parser)
