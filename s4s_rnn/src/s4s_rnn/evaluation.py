import os
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from sweat4science.messages import Session
from s4s_rnn import utils


_MINMAX = MinMaxScaler.__name__
_STANDARD = utils.Standardization.__name__


class ExperimentEvalutation(object):
    def __init__(self, session):
        self.session = session
        self.predictions = {}
        self.mse = {}
        self._scalers = {}
        self.weight_files = {}

        data_x, data_y = utils.get_data_from_sessions([session], num_timesteps=None, output_dim=1,
                                                      normalize=False, return_norm=False, old_norm=False)
        data = np.append(data_x, data_y, axis=1)
        num_feature = data_x.shape[1]
        scaler_min_max, data_min_max = utils.get_scaler(data, old_norm=False, return_data=True)
        scaler_standardization, data_standard = utils.get_scaler(data, old_norm=True, return_data=True)
        self.add_scaler(scaler_min_max)
        self.add_scaler(scaler_standardization)
        self._x_normed = {_MINMAX : data_min_max[:, :num_feature], _STANDARD : data_standard[:, :num_feature]}
        self._true_output = data_y
        pass

    def add_scaler(self, new_scaler):
        """
        :param new_scaler:
        :return: None
        """
        scaler_key = new_scaler.__class__.__name__
        if scaler_key in self._scalers:
            print("add_scaler: scaler %s already recorded" % scaler_key)
            return
        self._scalers[scaler_key] = new_scaler
        return

    def update_true_output(self, true_outputs, unnormalize=True, old_norm=False):
        """

        :param true_outputs:
        :param unnormalize:
        :param old_norm:
        :return:
        """
        if unnormalize:
            scaler_key = _STANDARD if old_norm else _MINMAX
            if scaler_key not in self._scalers:
                print('update_true_output: %s scaler is not initialized' % scaler_key)
                return
            self._true_output = utils.unnormalize(true_outputs, self._scalers[scaler_key])
        else:
            self._true_output = true_outputs
        return

    def add_prediction(self, name, prediction, unnormalize=True, old_norm=False):
        """

        :param name:
        :param prediction:
        :param unnormalize:
        :param old_norm:
        :return:
        """
        scaler_key = _STANDARD if old_norm else _MINMAX
        if scaler_key not in self._scalers:
            print('add_prediction: %s scaler is not initialized' % scaler_key)
            return

        if name in self.predictions:
            print("add_prediction: %s already in predictions" % name)
            return

        if unnormalize:
            self.predictions[name] = utils.unnormalize(prediction, self._scalers[scaler_key])
        else:
            self.predictions[name] = prediction
            pass

        self.mse[name] = np.mean((self.predictions[name] - self._true_output[-len(prediction):])**2)
        return

    def evaluate(self, model_json, prediction_name, weight_file=None, time_horizon=None):
        match = re.match('(?:lstm|gru)_lookback(\d+)_\d+neurons(.*)', prediction_name)
        if match is None:
            print("%s evaluate: invalid prediction_name %s" % (ExperimentEvalutation.__name__, prediction_name))
            return
        if len(match.group(2)) == 0:
            old_norm = False
        else:
            old_norm = True
            pass
        num_tsteps = int(match.group(1))

        if prediction_name not in self.weight_files:
            if weight_file is None:
                print("No weight file associated with key %s" % (prediction_name))
                return
            self.weight_files[prediction_name] = weight_file
            pass

        from keras.models import model_from_json
        model = model_from_json(model_json)

        scaler_key = _STANDARD if old_norm else _MINMAX
        data_x = utils.reshape_array_by_time_steps(self._x_normed[scaler_key], time_steps=num_tsteps)
        data_y = utils.normalize_with_scaler(self._true_output, self._scalers[scaler_key])

        prediction = utils.evaluate_model(model, self.weight_files[prediction_name],
                                          data_x, data_y, horizon=time_horizon)

        if time_horizon is None:
            prediction_key = prediction_name
        else:
            prediction_key = prediction_name + ("_%dhorizon" % time_horizon)
            pass

        self.add_prediction(prediction_key, prediction)

        return

    def plot_predictions(self, prediction_names, plot_tiltle, file_name=None):
        """

        :param prediction_names:
        :param plot_tiltle:
        :param file_name:
        :return:
        """
        for name in prediction_names:
            if name not in self.predictions:
                print("plot_predictions: unknown experiement name - %s" % name)
                pass
            pass

        predictions = list(map(self.predictions.get, prediction_names))
        utils.plot_predictions(predictions, prediction_names, self._true_output,
                               plot_tiltle, file_name=file_name)
        return

    pass


class ExperimentEvalutationDict(dict):
    """
    Class to store ExperimentEvalutation objects as dictionary
    """
    def __init__(self, sessions=None):
        super().__init__()
        self.model_json = {}
        self.sessions = {}
        if sessions is None:
            return

        for session in iter(sessions):
            if session.__class__.__name__ != 'Session':
                raise ValueError("%s: expected list of %s, got list of %s" % (ExperimentEvalutationDict.__name__, 'Session',
                                                                              Session.__name__))
                pass

            self.add_session(session)

            self.update_experiment(session)

            pass
        return

    def add_session(self, session):
        self[self.get_session_key(session.name)] = ExperimentEvalutation(session=session)
        return

    def get_session_key(self, session_name):
        match = re.match('.+/running_indoor_(.+)/(\d+)', session_name)
        return "%s_%s" % (match.groups()[0], match.groups()[1])

    def get_prediction_name(self, model_name, num_tstep, num_neuron, old_norm):
        prediction_name = "%s_lookback%s_%sneurons" % (model_name, num_tstep, num_neuron)
        if old_norm:
            prediction_name += "_oldnorm"
            pass
        return prediction_name

    def update_experiment(self, exp_eval):
        #TODO: check prediction_name in self.model_json
        return

    def add_model_json(self, model_file_path, update=False, old_norm=False):
        base_name = os.path.basename(model_file_path)
        match = re.match('(gru|lstm)_\w+_(\d{8})_(\d{2})step_(\d{2})in_(\d{3})hidden_(\d{3})epoch_model.json',
                         base_name)
        if match is None:
            print("%s: Invalid model file name: %s" % (ExperimentEvalutationDict.__name__, base_name))
            return False

        model_type = match.group(1)
        date_string = match.group(2)
        num_tstep = match.group(3)
        num_input = match.group(4)
        num_neuron = match.group(5)
        num_epoch = match.group(6)
        # print("%s_%s_%s_%s_%s_%s" % (model_type, date_string, num_tstep, num_input, num_neuron, num_epoch))
        prediction_name = self.get_prediction_name(model_type, num_tstep, num_neuron, old_norm)

        if prediction_name in self.model_json\
                and self.model_json[prediction_name] is not None\
                and not update:
            return False

        try:
            json_file = open(model_file_path, 'r')
        except IOError as e:
            print("%s: IOError caught when opening %s:\n%s" % (ExperimentEvalutationDict.__name__, base_name, e))
            return False
        else:
            loaded_model_json = json_file.read()
            json_file.close()
            pass

        self.model_json[prediction_name] = loaded_model_json
        return True

    def add_weight_file(self, weight_file_name, old_norm=False):
        base_name = os.path.basename(weight_file_name)
        if not os.path.exists(weight_file_name):
            print("%s: file doesn't exist: %s" % (ExperimentEvalutationDict.__name__, weight_file_name))
            return

        match = re.match('((gru|lstm)_\w+_(\d{8})_(\d{2})step_(\d{2})in_(\d{3})hidden_(\d{3})epoch)_(\S+)_weights.h5',
                         base_name)
        if match is None:
            print("%s: Invalid weight file name: %s" % (ExperimentEvalutationDict.__name__, base_name))
            return

        model_type = match.group(2)
        date_string = match.group(3)
        num_tstep = match.group(4)
        num_input = match.group(5)
        num_neuron = match.group(6)
        num_epoch = match.group(7)
        session_name = match.group(8)
        # print("%s_%s_%s_%s_%s_%s_%s" % (model_type, date_string, num_tstep, num_input, num_neuron, num_epoch, session_name))
        if session_name not in self:
            print("%s: session %s not added" % (ExperimentEvalutationDict.__name__, session_name))
            return

        prediction_name = self.get_prediction_name(model_type, num_tstep, num_neuron, old_norm)
        if prediction_name not in self.model_json or self.model_json[prediction_name] is None:
            dir_name = os.path.dirname(weight_file_name)
            model_file_name = os.path.join(dir_name, match.group(1) + "_model.json")
            print("%s: model for %s not recorded, looking for %s in %s" % (ExperimentEvalutationDict.__name__,
                                                                           model_file_name, prediction_name, dir_name))
            if not self.add_model_json(model_file_name):
                print("%s: can't add model JSON from %s" % (ExperimentEvalutationDict.__name__, model_file_name))
                return
            pass

        self[session_name].weight_files[prediction_name] = weight_file_name

        return

    pass
