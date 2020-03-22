import numpy as np
import scipy.stats


class Quantization:
    """
        Help class that handles the quantization of the magnitude involved in the problem
    """

    def __init__(self, vars_ls, func):
        """
             Initiates a list which keeps the bins intervals for each variable of the problem

        :param vars_ls: An iterable which contains tuple entries (start, stop, num_bins) for each variable
        """

        self.func = func  # TODO it seems func is not needed as we can set freq=1 if we want to exclude one dimension
        vars_ls = self.func(vars_ls)
        self.vars_bins = [self.initiate_var(*var) for var in vars_ls]
        self.dimensions = [len(var_bin) - 1 for var_bin in self.vars_bins]  # num of bins for each variable

    @staticmethod
    def initiate_var(start, stop, num_bins):
        """
            Splits the interval of possible values that a magnitude can take into bins

        :param start:
        :param stop:
        :param num_bins:
        :return: an array that contains the numbers that constitute bins bounds
        """

        # TODO investigate if it is useful to split the interval based on a distribution
        # so that higher bin frequency is used for points which are more probable to come up

        intervals = np.linspace(start, stop, num_bins + 1, endpoint=True)
        # intervals = scipy.stats.norm.ppf(intervals)
        return intervals

    def digitize(self, observations):
        """

        :param observations: list that contains continues values foreach variable of the problem
        :return: their quantized equivalents
        """

        observations = self.func(observations)
        assert len(observations) == len(self.vars_bins)  # validate that there are observations for every variable
        digitized = tuple(
            np.digitize(observation, var_bin) - 1 for observation, var_bin in zip(observations, self.vars_bins))

        # TODO check for values that fall outside of bins borders
        # If values in observations are beyond the bounds of bins, 0 or len(bins) is returned as appropriate.
        return digitized


if __name__ == 'main':
    pass
