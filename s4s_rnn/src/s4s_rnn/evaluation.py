from s4s_rnn import utils


class ExperimentEvalutation(object):
    def __init__(self, exp_name, subject_name, scaler):
        self._true_outputs = None
        self.experiment_name = exp_name
        self.subject_name = subject_name
        self._scaler = scaler
        self.predictions = {}
        self.mse = {}
        pass

    def update_scaler(self, new_scaler):
        if new_scaler is None:
            raise ValueError("New scaler cannot be None")
        self._scaler = new_scaler
        return

    def update_true_output(self, true_outputs, unnormalize):
        if unnormalize:
            self._true_outputs = utils.unnormalize(true_outputs, self._scaler)
        else:
            self._true_outputs = true_outputs
        return

    def add_predictions(self, name, prediction, unnormalize=True):
        if self._scaler is None:
            print("Scaler for normalization is not updated")
            return
        if unnormalize:
            self.predictions[name] = utils.unnormalize(prediction, self._scaler)
        return

    pass
