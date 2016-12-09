import numpy as np
from s4s_rnn import utils


class ExperimentEvalutation(object):
    def __init__(self, exp_name, scaler, true_output, unnormalize=True):
        self.experiment_name = exp_name
        self._scaler = scaler
        if unnormalize:
            self.true_output = utils.unnormalize(true_output, self._scaler)
        else:
            self.true_output = true_output
            pass
        self.predictions = {}
        self.mse = {}
        pass

    def update_scaler(self, new_scaler):
        if new_scaler is None:
            raise ValueError("New scaler cannot be None")
        self._scaler = new_scaler
        return

    def update_true_output(self, true_outputs, unnormalize=True):
        if unnormalize:
            self.true_output = utils.unnormalize(true_outputs, self._scaler)
        else:
            self.true_output = true_outputs
        return

    def add_prediction(self, name, prediction, unnormalize=True):
        if self._scaler is None:
            print("Scaler for normalization is not updated")
            return

        if name in self.predictions:
            print("%s already in predictions" % name)
            return

        if unnormalize:
            self.predictions[name] = utils.unnormalize(prediction, self._scaler)
        else:
            self.predictions[name] = prediction
            pass

        self.mse[name] = np.mean((self.predictions[name]
                                  - self.true_output[-len(prediction):])**2)
        return

    pass
