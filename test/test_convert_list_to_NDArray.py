
import unittest

import numpy as np
from mlexpt.utils.core import convert_listnum_ndarray


class TestConversion(unittest.TestCase):
    def test_ConversionFlattenList(self):
        list1 = [3., 4., 5.]
        np.testing.assert_array_equal(convert_listnum_ndarray(list1), np.array([3., 4., 5.]))

    def test_ConversionRank2Matrix(self):
        list2 = [[3., 4., 5.], [-1., -2., 5.]]
        np.testing.assert_array_equal(convert_listnum_ndarray(list2),
                                      np.array([[3., 4., 5.], [-1., -2., 5.]]))



if __name__ == '__main__':
    unittest.main()

