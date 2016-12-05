#!/usr/bin/env python

import os
import re
import time
import argparse
import textwrap


def get_arguments():
    parser = argparse.ArgumentParser(description=textwrap.dedent('''\
        Script to train RNN models for the Sweat4Science data set
        '''))
    parser.add_argument('num_epoch', type=int,
                        help='number of epochs to run training')
    parser.add_argument('--model', '-m', type=str, choices=['lstm', 'gru'], default='lstm',
                        help='RNN model to train')
    return parser.parse_args()


def main(arguments):
    import numpy as np
    from sklearn.model_selection import KFold
    from keras.models import model_from_json
    from keras.callbacks import CSVLogger, ReduceLROnPlateau

    from sweat4science.workspace.Workspace import Workspace
    from sweat4science.evaluation.sessionset import MF_sessionset as mfs
    from sweat4science import s4sconfig

    from s4s_rnn import models, utils

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

    print("\nconstructing LSTM model...")
    input_dim = 4
    output_dim = 1
    hidden_neurons = 400
    date_string = time.strftime("%Y%m%d")

    # Cross validation training
    kf = KFold(len(sessions))

    for ntsteps in [5, 10, 15]:
        model = models.create_model('lstm', hidden_neurons, input_dim=None,
                                    input_shape=(ntsteps, input_dim),
                                    output_dim=output_dim)
        # Construct meaningful base name
        base_name = "%s_indoor_%s_%02dstep_%02din_%03dhidden_%03depoch_" \
                    % (arguments.model, date_string, ntsteps, input_dim,
                       hidden_neurons, arguments.num_epoch)
        base_name = os.path.join("train_results", base_name)

        # serialize model to JSON
        model_json = model.to_json()
        model_file_name = base_name + "model.json"
        with open(model_file_name, "w") as json_file:
            json_file.write(model_json)
            pass
        print("\n----------------------------------------------------\n")
        print("Base file name: %s" % (base_name))
        print("Looking back %d time steps\n" % (ntsteps))
        for train_index, test_index in kf.split(sessions):
            print("\n--------------------------\n")
            train_sessions = sessions[train_index]
            test_sessions = sessions[test_index]
            print("Training on:\n" + "\n".join(map(str, train_sessions)))
            print("\nTesting on:\n" + "\n".join(map(str, test_sessions)))

            print("\nLoading model from: " + model_file_name)
            json_file = open(model_file_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop')

            match = re.match('.+/running_indoor_(.+)/(\d+)>', str(test_sessions[0]))
            # put name of evaluation set in saved filenames
            cross_validation_name = base_name + match.groups()[0] + "_" + match.groups()[1] + "_"

            train_data_x, train_data_y = utils.get_data_from_sessions(train_sessions, ntsteps)
            test_data_x, test_data_y = utils.get_data_from_sessions(test_sessions, ntsteps)

            csv_logger = CSVLogger(cross_validation_name + "training.log", append=False)
            #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=0.001)
            loaded_model.fit(train_data_x, train_data_y, batch_size=(1),
                             nb_epoch=arguments.num_epoch, validation_data=(test_data_x, test_data_y),
                             callbacks=[csv_logger], verbose=2)

            # serialize weights to HDF5
            loaded_model.save_weights(cross_validation_name + "weights.h5")
            print("Saved model to disk")

            pass
        pass
    return

if __name__ == "__main__":
    arguments = get_arguments()
    main(arguments)
    pass

