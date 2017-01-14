#!/usr/bin/env python3

import os
import re
import time
import argparse
import textwrap

from keras.models import model_from_json
from keras.callbacks import CSVLogger

from s4s_rnn import models, utils

_LOOKBACKS = [5, 10, 15]
_NEURON_NUMS = list(range(10, 100, 10))
_INPUT_DIM = 4
_OUTPUT_DIM = 1
_DEFAULT_LOOKBACK = 10


def get_arguments():
    parser = argparse.ArgumentParser(description=textwrap.dedent('''\
        Script to train RNN models for the Sweat4Science data set
        '''))
    parser.add_argument('num_epoch', type=int,
                        help='number of epochs to run training')
    parser.add_argument('--model', '-m', type=str,
                        choices=['lstm', 'gru', 'rnn'], default='lstm',
                        help='RNN model to train')
    parser.add_argument('--scenario', '-s', type=str,
                        choices=['vary_time', 'vary_neuron'],
                        default='vary_time',
                        help='training scenarios to run')
    parser.add_argument('--num_neurons', '-n', type=int, default=10,
                        help='number of hidden neurons in the perceptron layer')
    parser.add_argument('--multi_process', '-p', type=bool, default=False,
                        help='number of hidden neurons in the perceptron layer')
    return parser.parse_args()


def train_keras_model(train_sessions, test_sessions, num_tsteps, base_name, verbose=2):
    json_file = open(base_name + '_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop')

    match = re.match('.+/running_indoor_(.+)/(\d+)>', str(test_sessions[0]))
    # put name of evaluation set in saved filenames
    cross_validation_name = "%s_%s_%s" % (base_name, match.groups()[0], match.groups()[1])

    train_data_x, train_data_y = utils.get_data_from_sessions(train_sessions, num_tsteps)
    test_data_x, test_data_y = utils.get_data_from_sessions(test_sessions, num_tsteps)

    csv_logger = CSVLogger(cross_validation_name + "_training.log", append=False)
    loaded_model.fit(train_data_x, train_data_y, batch_size=(1),
                     nb_epoch=arguments.num_epoch, validation_data=(test_data_x, test_data_y),
                     callbacks=[csv_logger], verbose=verbose)

    # serialize weights to HDF5
    loaded_model.save_weights(cross_validation_name + "_weights.h5")

    print("Completed training for session %s" % test_sessions[0].name)
    return


def train_cross_validation(sessions, model, num_hidden, num_tsteps, date_string):
    from sklearn.model_selection import KFold

    model = models.create_model(model, num_hidden, input_dim=None,
                                input_shape=(num_tsteps, _INPUT_DIM),
                                output_dim=_OUTPUT_DIM)
    # Construct meaningful base name
    base_name = "%s_indoor_%s_%02dstep_%02din_%03dhidden_%03depoch" \
                % (arguments.model, date_string, num_tsteps, _INPUT_DIM,
                   num_hidden, arguments.num_epoch)
    # print("Base file name: %s" % (base_name))
    base_name = os.path.join("train_results", base_name)

    # serialize model to JSON
    model_json = model.to_json()
    model_file_name = base_name + "_model.json"
    with open(model_file_name, "w") as json_file:
        json_file.write(model_json)
        pass

    kf = KFold(len(sessions))
    func_args = []
    for train_index, test_index in kf.split(sessions):
        train_sessions = sessions[train_index]
        test_sessions = sessions[test_index]
        if arguments.multi_process:
            func_args.append([train_sessions, test_sessions, num_tsteps, base_name, 0])
        else:
            print("\n--------------------------")
            print("Training on:\n" + "\n".join(map(str, train_sessions)))
            print("Testing on:\n" + "\n".join(map(str, test_sessions)))

            print("\nLoading model from: " + model_file_name)
            train_keras_model(train_sessions, test_sessions, num_tsteps, base_name)
            print("Saved model to disk")
            pass

        pass
    if arguments.multi_process:
        import sweat4science as s4s
        print("Starting analysis for " + str(len(func_args)) + " processes")
        results_handle = s4s.execute_parallel(train_keras_model, func_args)
        s4s.wait_for_results(results_handle)
        pass

    return


def main(arguments):
    import numpy as np

    from sweat4science.workspace.Workspace import Workspace
    from sweat4science.evaluation.sessionset import MF_sessionset as mfs
    from sweat4science import s4sconfig

    workspace_folder = os.path.join(s4sconfig.workspace_dir, "session-data")
    ws = Workspace(workspace_folder)

    sessions = mfs.ICT_indoor(ws)

    # Removing "slope" sessions
    for session in sessions:
        if len(re.findall("slope", str(session))) > 0:
            sessions.remove(session)
        pass
    sessions = np.array(sessions)

    print("Using sessions: ")
    print("\n".join(map(str, sessions)))

    print("\nconstructing %s model..." % str.upper(arguments.model))
    date_string = time.strftime("%Y%m%d")
    print("----------------------------------------------------\n"
          "Model: %s\n" % arguments.model)
    if arguments.scenario == 'vary_time':
        print("Number of hidden neurons: %d\n"
              "Lookback: %s time steps\n"
              "----------------------------------------------------\n"
              % (arguments.num_neurons, str(_LOOKBACKS)))
        for lookback in _LOOKBACKS:
            print("\n------------------------------------------")
            print("Looking back %d time steps\n" % (lookback))

            train_cross_validation(sessions, arguments.model, arguments.num_neurons,
                                   lookback, date_string)
            pass
    elif arguments.scenario == 'vary_neuron':
        print("Number of hidden neurons: %s\n"
              "Lookback: %d time steps\n"
              "----------------------------------------------------\n"
              % (str(_NEURON_NUMS), _DEFAULT_LOOKBACK))
        for num_neuron in _NEURON_NUMS:
            print("\n------------------------------------------")
            print("Trainning with %d neurons\n" % (num_neuron))

            train_cross_validation(sessions, arguments.model, num_neuron,
                                   _DEFAULT_LOOKBACK, date_string)
            pass
    else:
        print("Unsupported scenario: %s" % arguments.scenario)
        pass
    return

if __name__ == "__main__":
    arguments = get_arguments()
    main(arguments)
    pass

