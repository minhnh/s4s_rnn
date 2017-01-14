#!/usr/bin/env python3
import argparse
import glob
import os
import pickle
import re
import textwrap

from s4s_rnn import utils, evaluation


def get_arguments():
    parser = argparse.ArgumentParser(description=textwrap.dedent('''
        Script to evaluate and save predictions
        '''))
    parser.add_argument('file_out', type=argparse.FileType('wb'),
                        help="File to store ExperimentEvalutationDict object")
    parser.add_argument('result_dir', action=utils.ReadableDir,
                        help="Directory with model json's and weight files")
    parser.add_argument('--time_horizons', '-t', action='store_true',
                        help="If true evaluate at 10-60s time horizons")
    return parser


def main(parser):
    arguments = parser.parse_args()

    sessions = utils.get_sessions()
    print("Evaluating on %d sessions:" % len(sessions))
    print("\t\n".join(map(str, sessions)))
    eval_dict = evaluation.ExperimentEvalutationDict(sessions)
    # load models
    for model_path in glob.glob(os.path.join(arguments.result_dir, "*.json")):
        match = re.match('.+20161114.+', model_path)
        old_norm = False
        if match is not None:
            old_norm = True
            pass
        eval_dict.add_model_json(model_path, old_norm=old_norm)
        pass
    # load weights
    for weight_path in glob.glob(os.path.join(arguments.result_dir, "*.h5")):
        match = re.match('.+20161114.+', weight_path)
        old_norm = False
        if match is not None:
            old_norm = True
            pass
        eval_dict.add_weight_file(weight_path, old_norm=old_norm)
        pass
    # Evaluate
    for regressor_size in [5, 10, 15]:
        eval_dict.evaluate_old_model("SVM", regressor_size)
        pass

    eval_dict.evaluate()

    if arguments.time_horizons:
        print("----------------------------------------------------------------------------\n"
              "Evaluating predictions at 10s-60s horizons")
        prediction_key_list = []
        for prediction_key in eval_dict.model_json:
            _, num_neuron, _, old_norm, _ = evaluation.parse_prediction_key(prediction_key)
            if num_neuron == 10 and not old_norm:
                prediction_key_list.append(prediction_key)
                pass
            pass
        print("\n".join(prediction_key_list))
        for horizon in range(1, 7):
            print("--------------------------------------\n"
                  "Evaluating %ds horizon" % (horizon*10))
            eval_dict.evaluate(prediction_key_list, time_horizon=horizon)
            pass
        pass
    # Save to file_out
    print("Saving evaluation results to %s" % arguments.file_out.name)
    pickle.dump(eval_dict, arguments.file_out, pickle.HIGHEST_PROTOCOL)

    return


if __name__ == "__main__":
    parser = get_arguments()
    main(parser)
