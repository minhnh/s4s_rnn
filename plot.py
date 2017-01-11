#!/usr/bin/env python3
import os
import argparse
import textwrap


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


def main(parser):
    arguments = parser.parse_args()

    if arguments.scenario == 'hbm_velocity':
        workspace_folder = os.path.join(workspace_dir, "session-data")
        ws = Workspace(workspace_folder)
        sessions = mfs.ICT_indoor(ws)
        eval_dict = evaluation.ExperimentEvalutationDict(sessions)
        plotting.plot_hbm_velocity(eval_dict, ['session_01_20150219', 'session_03_20150305', 'lactate_test_20150408'])
    elif arguments.scenario == 'epoch_error':
        if arguments.file is None:
            print("epoch_error scenario requires argument log file")
            parser.print_usage()
            return
        plotting.plot_epoch(arguments.file)
        pass
    return


if __name__ == "__main__":
    parser = get_arguments()
    main(parser)
