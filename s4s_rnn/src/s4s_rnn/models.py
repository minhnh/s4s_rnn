import  numpy
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM, GRU

import sweat4science as s4s


def create_model(model_name, hidden_neurons, input_dim, output_dim, input_shape=None):
    """

    :param model_name:
    :param hidden_neurons:
    :param input_dim:
    :param output_dim:
    :param input_shape:
    :return:
    """
    model = Sequential()
    if model_name == 'lstm':
        rnn_layer = LSTM
        pass
    elif model_name == 'gru':
        rnn_layer = GRU
        pass
    else:
        return None

    if hidden_neurons == 0:
        hidden_neurons = output_dim
        pass

    if input_shape is None:
        model.add(rnn_layer(hidden_neurons, input_dim=input_dim, return_sequences=False))
        pass
    else:
        model.add(rnn_layer(hidden_neurons, input_shape=input_shape, return_sequences=False))
        pass

    if not hidden_neurons == 0:
        model.add(Dense(output_dim, input_dim=hidden_neurons))
        pass
    model.add(Activation('linear'))
    #model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model


def run_old_models(model_definition, regressor_size, sessions, prediction_horizon=None, multi_process="processes", run_in_parallel=True):
    """

    :param model_definition:
    :param regressor_size:
    :param sessions:
    :param prediction_horizon:
    :param multi_process: "processes", "threads" or "cluster"
    :param run_in_parallel:
    :return:
    """
    from sweat4science.evaluation.ga import InputGnome, InputGnomePart
    from sweat4science.model.optimization.implemention.ModelInput import BiasModelInput
    from sweat4science.model.optimization.implemention.modelinput.stdinput \
        import HBMModelInput, AccelerationModelInput, SlopeModelInput, DistanceModelInput, VelocityModelInput

    # select how to do multi processing
    s4s.s4sconfig.execution_engine = multi_process

    regressors = [regressor_size]
    model_input_gnome = InputGnome()
    model_input_gnome.append(InputGnomePart("bias", BiasModelInput, [1], True))
    model_input_gnome.append(InputGnomePart("HR", HBMModelInput, [1], True))
    model_input_gnome.append(InputGnomePart("distance", DistanceModelInput, regressors, True))
    model_input_gnome.append(InputGnomePart("velocity", VelocityModelInput, regressors, True))
    model_input_gnome.append(InputGnomePart("slope", SlopeModelInput, regressors, True))
    model_input_gnome.append(InputGnomePart("acceleration", AccelerationModelInput, regressors, True))

    print("Using model input: " + str(model_input_gnome))

    results = {}
    if run_in_parallel:
        func_args = []
        for s in sessions:
            train_sessions = [ss for ss in sessions if ss.name != s.name]
            test_sessions = [s]
            func_args.append([train_sessions, test_sessions, prediction_horizon, model_definition, model_input_gnome])
            pass

        print("Starting analysis for " + str(len(func_args)) + " processes")
        results_handle = s4s.execute_parallel(run_analysis_old_models, func_args)

        s4s.wait_for_results_callback(results_handle, results.update)

    else:
        train_sessions = sessions[:-1]
        test_sessions = sessions[-1:]
        prediction = run_analysis_old_models(train_sessions, test_sessions, prediction_horizon, model_definition, model_input_gnome)
        results[test_sessions[0].name] = [prediction, test_sessions[0].hbm]
        pass

    predictions = []
    actual_hbms = []
    for session_name in results:
        predictions.append(results[session_name][0])
        actual_hbms.append(results[session_name][1])
        pass

    p = numpy.hstack(predictions)
    h = numpy.hstack(actual_hbms)

    print("Resulting RMSE: " + str(numpy.sqrt(((p - h) ** 2).mean())))
    return results


def run_analysis_old_models(train_sessions, test_sessions, prediction_horizon, model_definition, current_gnome):
    """

    :param train_sessions:
    :param test_sessions:
    :param prediction_horizon:
    :param model_definition:
    :param current_gnome:
    :return:
    """
    from sweat4science.model.optimization.implemention.ModelError import StdModelError

    #creating the instances
    model_input = current_gnome.create()
    ts_blueprint = s4s.define(model_definition,model_input, StdModelError())
    ts_current = s4s.create_instance(ts_blueprint)

    # fitting process
    s4s.fit(train_sessions,**ts_current)
    #eval_results = evaluate_error(test, test, [6], **ts_current)

    # evaluation
    if prediction_horizon is None:
        prediction = s4s.simulate_session(test_sessions[0], **ts_current)
    else:
        prediction = s4s.predict_session(prediction_horizon, test_sessions[0], **ts_current)
        pass

    return {test_sessions[0].name : [prediction, test_sessions[0].hbm]}
