import numpy as np
from s4s_rnn import utils
from sklearn.preprocessing import MinMaxScaler


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

        self.mse[name] = np.mean((self.predictions[name]
                                  - self.true_output[-len(prediction):])**2)
        return

    pass
