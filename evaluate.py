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
    eval_dict.evaluate(list(eval_dict.model_json.keys()))
    # Save to file_out
    print("Saving evaluation results to %s" % arguments.file_out.name)
    pickle.dump(eval_dict, arguments.file_out, pickle.HIGHEST_PROTOCOL)

    return


if __name__ == "__main__":
    parser = get_arguments()
    main(parser)
