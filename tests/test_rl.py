import unittest
import gym
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

    def test_quantization(self):
        """  """

        self.assertEqual(self.quantizator.dimensions, self.var_freq)
        # error case out of bin
        test_obs = [self.high_intervals[0] + 1, -0.1, self.high_intervals[2] + 1, 0.1]
        self.assertEqual((self.var_freq[0], 0, self.var_freq[2], 1), self.quantizator.digitize(test_obs))




if __name__ == '__main__':
    unittest.main()
