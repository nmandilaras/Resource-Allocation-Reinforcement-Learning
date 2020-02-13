import unittest
from quantization import Quantization


class TestStringMethods(unittest.TestCase):

    def test_quantization(self):
        """  """
        vars_ls = [(1, 5, 3), (10, 19, 20)]
        quantizator = Quantization(vars_ls)

        print(quantizator.vars_bins)
        self.assertEqual(quantizator.dimensions, [3, 20])
        # self.assertEqual(quantizator.digitize((3, 11)), (2, 2))

    def test_lambda(self):
        func = lambda x: x[2:]
        ls = [1, 2, 3, 4]

        self.assertEqual(func(ls), [3, 4])

if __name__ == '__main__':
    unittest.main()
