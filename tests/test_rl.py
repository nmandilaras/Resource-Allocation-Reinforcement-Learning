import unittest
import gym
import numpy as np
from utils import constants
from utils.quantization import Quantization


# logging.config.fileConfig('logging.conf')
# logger = logging.getLogger('simpleExample')


class TestRL(unittest.TestCase):
    env = gym.make(constants.environment)
    high_intervals = env.observation_space.high
    low_intervals = env.observation_space.low
    var_freq = [4, 2, 6, 2]
    vars_ls = list(zip(low_intervals, high_intervals, var_freq))
    quantizator = Quantization(vars_ls, lambda x: x)

    def test_running_mean(self):
        """  """
        N = 10
        d = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'k': 10, 'm': 11, 'l': 12}
        # res = np.convolve(list(d.values()), np.ones((N,)) / N, mode='valid')
        res = sum(list(d.values())[-N:]) / N
        print(res)

    def test_quantization(self):
        """  """

        self.assertEqual(self.quantizator.dimensions, self.var_freq)
        # error case out of bin
        test_obs = [self.high_intervals[0] + 1, -0.1, self.high_intervals[2] + 1, 0.1]
        self.assertEqual((self.var_freq[0], 0, self.var_freq[2], 1), self.quantizator.digitize(test_obs))


if __name__ == '__main__':
    unittest.main()
