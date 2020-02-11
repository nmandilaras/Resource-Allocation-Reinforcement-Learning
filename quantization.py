import numpy as np
import scipy.stats


class Quantization:

    def __init__(self, vars_dict):
        self.vars_bins = [self.initiate_var(*var) for var in vars_dict]

    @staticmethod
    def initiate_var(start, stop, num_bins):
        intervals = np.linspace(start, stop, num_bins + 1, endpoint=True)
        # intervals = scipy.stats.norm.ppf(intervals)
        return intervals

    def state_dims(self):
        return [len(var_bin) - 1 for var_bin in self.vars_bins]

    def digitize(self, observations):
        assert len(observations) == len(self.vars_bins)
        digitized = tuple(np.digitize(observation, var_bin) - 1 for observation, var_bin in
                     zip(observations, self.vars_bins))
        return digitized
