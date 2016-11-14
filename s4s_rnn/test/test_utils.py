#!/usr/bin/env python3

import os
import unittest
import numpy as np
from s4s_rnn import utils
from sweat4science import s4sconfig
from sweat4science.workspace.Workspace import Workspace


class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_reshape_array_by_time_steps_exact(self):
        test_input_array = np.arange(12).reshape(4, 3)
        ntstep = 2
        test_result_array = np.array(
            [[[0,  1,  2], [3,  4,  5]],
             [[3,  4,  5], [6,  7,  8]],
             [[6,  7,  8], [9, 10, 11]]])
        self.assertTrue(np.allclose(utils.reshape_array_by_time_steps(test_input_array, ntstep), test_result_array),
                        msg="Values of output array are wrong")
        pass

    def test_reshape_array_by_time_steps_shape(self):
        num_element = 24
        test_input_array = np.arange(num_element)
        for num_sample in [2, 3, 4, 6, 8]:
            input_dim = int(num_element / num_sample)
            test_input_array = test_input_array.reshape((num_sample, input_dim))
            for time_steps in range(1, num_sample):
                test_output = utils.reshape_array_by_time_steps(input_array=test_input_array, time_steps=time_steps)
                if test_output is None:
                    print("None returned for timte_steps: %d, input_dim: %d, num_samples %d"
                          % (time_steps, input_dim, num_sample))
                    pass
                else:
                    self.assertEqual(test_output.shape, (num_sample - time_steps + 1, time_steps, input_dim),
                                     msg="Dimensions of output arrays does not match with expected")
                    pass
                pass
            pass
        pass

    def test_reshape_array_by_time_steps_invalid_input(self):
        self.assertRaises(ValueError, utils.reshape_array_by_time_steps, np.arange(24).reshape(2, 3, 4))
        for wrong_time_steps in [-1, 0, 1.2, None]:
            self.assertRaises(ValueError, utils.reshape_array_by_time_steps,
                              np.arange(6).reshape(3, 2), time_steps=wrong_time_steps)
            pass
        self.assertEqual(utils.reshape_array_by_time_steps(np.arange(6).reshape(3, 2), time_steps=5).shape, (1, 3, 2),
                         msg="case of time_steps longer than number of samples not catched")
        pass

    def test_get_data_from_sessions(self):
        #TODO: test optional parameters
        workspace_folder = os.path.join(s4sconfig.workspace_dir, "session-data")
        ws = Workspace(workspace_folder)
        user_name = "MF83"
        experiment_name = ["running_indoor_lactate_test", "running_indoor_session_01"]
        session_number = None
        sessions = ws.get(user_name, experiment_name, session_number)
        for ntsteps in range(1, 10):
            data_x, data_y = utils.get_data_from_sessions(sessions, ntsteps)
            self.assertEqual(data_x.shape[2], 4, "data input dimension is not 4")
            self.assertEqual(data_x.shape[1], ntsteps, "second dimension of array should "
                                                       "equal number of time steps")
            self.assertEqual(data_y.shape[1], 1, "data output dimension is not 1")
            self.assertEqual(data_x.shape[0], data_y.shape[0], "number of samples do not match")
            pass
        pass

    pass


if __name__ == '__main__':
    unittest.main()
    pass
