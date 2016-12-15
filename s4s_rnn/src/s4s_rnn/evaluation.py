import os
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from s4s_rnn import utils


_MINMAX = MinMaxScaler.__name__
_STANDARD = utils.Standardization.__name__


class ExperimentEvalutation(object):
    def __init__(self, exp_name, scaler, true_output, unnormalize=True):
        self.experiment_name = exp_name

        self._scalers = {}
        if scaler is not None:
            self._scalers[scaler.__class__.__name__] = scaler
            pass

        if unnormalize:
            self.true_output = utils.unnormalize(true_output, scaler)
        else:
            self.true_output = true_output
            pass
        self.predictions = {}
        self.mse = {}
        pass

    def add_scaler(self, new_scaler):
        if new_scaler is None:
            raise ValueError("add_scaler: New scaler cannot be None")

        scaler_key = new_scaler.__class__.__name__
        if scaler_key in self._scalers:
            print("add_scaler: scaler %s already recorded" % scaler_key)
            return
        self._scalers[scaler_key] = new_scaler
        return

    def update_true_output(self, true_outputs, unnormalize=True, old_norm=False):
        if unnormalize:
            scaler_key = _STANDARD if old_norm else _MINMAX
            if scaler_key not in self._scalers:
                print('update_true_output: %s scaler is not initialized' % scaler_key)
                return
            self.true_output = utils.unnormalize(true_outputs, self._scalers[scaler_key])
        else:
            self.true_output = true_outputs
        return

    def add_prediction(self, name, prediction, unnormalize=True, old_norm=False):
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

        self.mse[name] = np.mean((self.predictions[name] - self.true_output[-len(prediction):])**2)
        return

    def plot_predictions(self, prediction_names, plot_tiltle, file_name=None):
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

    def __init__(self, evaluations=None):
        if evaluations is None:
            super().__init__()
            return

        super().__init__(evaluations)
        self.model_json = {}
        for key, value in iter(evaluations):
            if value.__class__.__name__ != ExperimentEvalutation.__name__:
                raise ValueError("%s: Expected %s, got: %s" % (self.__class__.__name__,
                                                               ExperimentEvalutation.__name__,
                                                               value.__class__.__name__))
                pass

            if key != value.experiment_name:
                raise ValueError("%s: key %s does not match name %s"
                                 % (self.__class__.__name__, key, value.experiment_name))
                pass

            self.update_experiment(value)

            pass
        return

    def update_experiment(self, exp_eval):
        #TODO: check prediction_name in self.model_json
        return

    def add_model_json(self, model_file_name, update=False):
        base_name = os.path.basename(model_file_name)
        match = re.match('(gru|lstm)_\w+_(\d{8})_(\d{2})step_(\d{2})in_(\d{3})hidden_(\d{3})epoch_model.json',
                         base_name)
        if match is None:
            print("%s: Invalid model file name: %s" % (self.__class__.__name__, base_name))
            return

        model_type = match.group(1)
        date_string = match.group(2)
        num_tstep = match.group(3)
        num_input = match.group(4)
        num_neuron = match.group(5)
        num_epoch = match.group(6)
        # print("%s_%s_%s_%s_%s_%s" % (model_type, date_string, num_tstep, num_input, num_neuron, num_epoch))
        prediction_name = "%s_lookback%s_%sneurons" % (model_type, num_tstep, num_neuron)

        if prediction_name in self.model_json\
                and self.model_json[prediction_name] is not None\
                and not update:
            return

        json_file = open(model_file_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model_json[prediction_name] = loaded_model_json
        return

    def add_experiment_from_file(self, file_name):
        return

    pass
