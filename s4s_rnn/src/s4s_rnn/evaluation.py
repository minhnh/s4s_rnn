import os
import re
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sweat4science.messages import Session
from s4s_rnn import utils


_MINMAX = MinMaxScaler.__name__
_STANDARD = utils.Standardization.__name__


def get_session_key(session_name):
    match = re.match('.+/running_indoor_(.+)/(\d+)', session_name)
    return "%s_%s" % (match.groups()[0], match.groups()[1])


def get_prediction_name(model_name, num_tstep, num_neuron, old_norm):
    prediction_name = "%s_lookback%s_%sneurons" % (model_name, num_tstep, num_neuron)
    if old_norm:
        prediction_name += "_oldnorm"
        pass
    return prediction_name


def parse_prediction_key(prediction_key):
    """
    :param prediction_key:
    :return:
    """
    match = re.match('(?:lstm|gru)_lookback(\d+)_\d+neurons(.*)', prediction_key)
    if match is None:
        return None, None, None

    num_tsteps = int(match.group(1))

    old_norm = False
    time_horizon = None
    if len(match.group(2)) != 0:
        match2 = re.match('(_oldnorm)?(_(\d+)horizon)?',  match.group(2))
        if match2.group(1) is not None:
            old_norm = True
            pass
        if match2.group(3) is not None:
            time_horizon = int(match2.group(3))
            pass
        pass

    return num_tsteps, old_norm, time_horizon


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
        self.true_output = data_y
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
            self.true_output = utils.unnormalize(true_outputs, self._scalers[scaler_key])
        else:
            self.true_output = true_outputs
        return

    def _add_prediction(self, name, prediction, old_norm, unnormalize=True):
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

        if unnormalize:
            self.predictions[name] = utils.unnormalize(prediction, self._scalers[scaler_key])
        else:
            self.predictions[name] = prediction
            pass

        self.mse[name] = np.mean((self.predictions[name] - self.true_output[-len(prediction):])**2)
        return

    def evaluate(self, model_json, prediction_key, weight_file=None):
        """
        Evaluate model and add prediction to self.predictions
        :param model_json:
        :param prediction_key:
        :param weight_file:
        :param time_horizon:
        :return: None
        """
        num_tsteps, old_norm, time_horizon = parse_prediction_key(prediction_key)
        if num_tsteps is None:
            print("%s evaluate: invalid prediction_name %s" % (ExperimentEvalutation.__name__, prediction_key))
            return

        if prediction_key not in self.weight_files:
            if weight_file is None:
                print("No weight file associated with key %s" % prediction_key)
                return
            self.weight_files[prediction_key] = weight_file
            pass

        if time_horizon is None:
            prediction_key_horizon = prediction_key
        else:
            prediction_key_horizon = prediction_key + ("_%dhorizon" % time_horizon)
            pass
        if prediction_key_horizon in self.predictions:
            print("evaluate: %s already in predictions" % prediction_key_horizon)
            return

        from keras.models import model_from_json
        model = model_from_json(model_json)

        scaler_key = _STANDARD if old_norm else _MINMAX
        data_x = utils.reshape_array_by_time_steps(self._x_normed[scaler_key], time_steps=num_tsteps)
        data_y = utils.normalize_with_scaler(self.true_output, self._scalers[scaler_key])

        prediction = utils.evaluate_model(model, self.weight_files[prediction_key],
                                          data_x, data_y, horizon=time_horizon)

        self._add_prediction(prediction_key_horizon, prediction, old_norm=old_norm)

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
        utils.plot_predictions(predictions, prediction_names, self.true_output,
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
        self.mse = {}
        self._true_outputs = None
        if sessions is None:
            return

        for session in iter(sessions):
            if session.__class__.__name__ != 'Session':
                raise ValueError("%s: expected list of %s, got list of %s" % (ExperimentEvalutationDict.__name__, 'Session',
                                                                              Session.__name__))
                pass

            self[get_session_key(session.name)] = ExperimentEvalutation(session=session)

            pass

        self._get_all_true_outputs()

        return

    def _get_all_true_outputs(self):
        """
        Add all true_output array to self._true_outputs
        """
        true_outputs = None
        for session_key in self.keys():
            true_outputs = self[session_key].true_output if true_outputs is None \
                else np.append(true_outputs, self[session_key].true_output, axis=0)
            pass
        self._true_outputs = true_outputs
        return

    def _get_all_predictions(self, prediction_key):
        """
        Add all prediction named with prediction_key to an array and return
        """
        predictions = None
        for session_key in self.keys():
            predictions = self[session_key].predictions[prediction_key] if predictions is None \
                else np.append(predictions, self[session_key].predictions[prediction_key], axis=0)
            pass
        return predictions

    def add_model_json(self, model_file_path, update=False, old_norm=False):
        """
        Add a valid model JSON to the model_json dictionary
        :param model_file_path: path to JSON file
        :param update: if true rewrite model_json
        :param old_norm: if True use standardization
        :return: None
        """
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
        prediction_name = get_prediction_name(model_type, num_tstep, num_neuron, old_norm)

        if prediction_name in self.model_json\
                and self.model_json[prediction_name] is not None\
                and not update:
            print("%s: not updating model_json entry %s with content from %s" % (
                ExperimentEvalutationDict.__name__, prediction_name, model_file_path))
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
        """
        Add a valid weight file to load when evaluating an ExperimentEvalutation object
        :param weight_file_name: name of keras weight file to be added
        :param old_norm: if True use old normalization
        :return: None
        """
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

        prediction_name = get_prediction_name(model_type, num_tstep, num_neuron, old_norm)
        if prediction_name not in self.model_json or self.model_json[prediction_name] is None:
            dir_name = os.path.dirname(weight_file_name)
            model_file_name = os.path.join(dir_name, match.group(1) + "_model.json")
            print("%s: model for %s not recorded, looking for %s in %s" % (ExperimentEvalutationDict.__name__,
                                                                           prediction_name, model_file_name, dir_name))
            if not self.add_model_json(model_file_name, old_norm=old_norm):
                print("%s: can't add model JSON from %s" % (ExperimentEvalutationDict.__name__, model_file_name))
                return
            pass

        self[session_name].weight_files[prediction_name] = weight_file_name

        return

    def evaluate(self, prediction_list=None, time_horizon=None):
        if prediction_list is None:
            prediction_list = self.model_json.keys()
            pass

        for prediction_name in prediction_list:
            print("Evaluating %s" % prediction_name)
            if prediction_name not in self.model_json:
                print("%s evaluate: unrecognized prediction %s" % (ExperimentEvalutationDict.__name__, prediction_name))
                continue
                pass

            if time_horizon is not None:
                prediction_key = prediction_name + ("_%dhorizon" % time_horizon)
            else:
                prediction_key = prediction_name

            for session_key in self:
                print("\ton %s" % session_key)
                self[session_key].evaluate(self.model_json[prediction_name], prediction_key=prediction_key)
                pass

            predictions = self._get_all_predictions(prediction_key)
            self.mse[prediction_key] = np.mean((predictions - self._true_outputs)**2)
            print("   MSE: %f" % self.mse[prediction_key])
            print("  RMSE: %f\n" % np.sqrt(self.mse[prediction_key]))

            pass
        return

    def _get_abs_errors(self, prediction_keys):
        abs_errors = []
        for key in prediction_keys:
            if key not in self.mse:
                print("plot_error_box_predictions: prediction %s is not evaluated" % key)
                continue
            else:
                prediction = self._get_all_predictions(key)
                abs_error = np.abs(prediction - self._true_outputs)
                # print("mse: %.2f" % np.mean(squared_error))
                abs_errors.append(abs_error)
                pass
            pass
        return abs_errors

    def plot_error_box_predictions(self, prediction_keys, title):
        """
        Styling ispired by http://matplotlib.org/examples/pylab_examples/boxplot_demo2.html
        :param prediction_keys:
        :param title:
        :return:
        """
        abs_errors = self._get_abs_errors(prediction_keys)
        utils.box_plot_error(abs_errors, title, labels=prediction_keys)
        return

    def plot_error_bar_predictions(self, title):
        abs_error_groups = []
        for ntstep in [5, 10, 15]:
            group = []
            for model_name in ['lstm', 'gru']:
                abs_errors = self._get_abs_errors(["%s_lookback%02d_400neurons" % (model_name, ntstep)])
                group.append(abs_errors[0])
                pass
            abs_error_groups.append(group)
            pass
        utils.bar_plot_error(abs_error_groups, title, ["Lookback 5", "Lookback 10", "Lookback 15"], ["LSTM", "GRU"])
        return

    pass
