# Copyright 2018 Side Li and Arun Kumar
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import scipy.sparse as sp
from numpy.testing import (
    run_module_suite, assert_equal
)

from morpheusflow.dataset import Dataset
from morpheusflow.data import NumPyData, SparseData, CSVData
from morpheusflow.utils import Converter

class TestNormalizedMatrix(object):
    def test_init(self):
        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1])]
        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]])]
        y = np.matrix([[1, 0, 1, 0, 1]]).transpose()

        dataset = Dataset(NumPyData(s), NumPyData(k), NumPyData(r), NumPyData(y))
        actual_x, actual_y = dataset.get_next()
        desired_x = np.array([[1.0, 2.0, 1.1, 2.2]])
        desired_y = np.array([1])
        assert_equal(actual_x, desired_x)
        assert_equal(actual_y, desired_y)

    def test_repeat(self):
        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1])]
        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]])]
        y = np.matrix([[1, 0, 1, 0, 1]]).transpose()

        dataset = Dataset(NumPyData(s), NumPyData(k), NumPyData(r), NumPyData(y)).repeat(2)
        for _ in range(10):
            dataset.get_next()

    def test_batch(self):
        s = np.matrix([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0], [9.0, 1.0]])
        k = [np.array([0, 1, 1, 0, 1])]
        r = [np.matrix([[1.1, 2.2], [3.3, 4.4]])]
        y = np.matrix([[1, 0, 1, 0, 1]]).transpose()

        dataset = Dataset(NumPyData(s), NumPyData(k), NumPyData(r), NumPyData(y)).batch(2)
        actual_x, actual_y = dataset.get_next()
        desired_x = np.array([[1.0, 2.0, 1.1, 2.2], [4.0, 3.0, 3.3, 4.4]])
        desired_y = np.array([[1], [0]])

        assert_equal(actual_x, desired_x)
        assert_equal(actual_y, desired_y)

    def test_sparse(self):
        s = sp.random(5, 3, density=1, format='coo')
        k = [np.array([0, 1, 1, 0, 1])]
        r = [sp.random(2, 2, density=1, format='csr')]
        y = np.matrix([[1, 0, 1, 0, 1]]).transpose()

        dataset = Dataset(SparseData(s), NumPyData(k), SparseData(r), NumPyData(y))
        actual_x, actual_y = dataset.get_next()
        desired_x = sp.hstack([s.tocsr()[[0]], r[0][[0]]])
        desired_y = np.array(y)[0]
        assert_equal(Converter.convert_sparse_tensor_to_coo(actual_x).toarray(), desired_x.toarray())
        assert_equal(actual_y, desired_y)

    def test_sparse_batch(self):
        s = sp.random(5, 2, density=1, format='coo')
        k = [np.array([0, 1, 1, 0, 1])]
        r = [sp.random(2, 2, density=1, format='csr')]
        y = np.matrix([[1, 0, 1, 0, 1]]).transpose()

        dataset = Dataset(SparseData(s), NumPyData(k), SparseData(r), NumPyData(y)).batch(2)
        actual_x, actual_y = dataset.get_next()
        desired_x = sp.hstack([s.tocsr()[[0, 1]], r[0][np.array([0, 1])]])
        desired_y = y[[0, 1]]
        assert_equal(Converter.convert_sparse_tensor_to_coo(actual_x).toarray(), desired_x.toarray())
        assert_equal(actual_y, desired_y)

    def test_csvread(self):
        test_csv = './test.csv'
        actual_csv = CSVData(test_csv).read()
        desired_csv = np.array([[1, 2, 3], [4, 5, 6]])
        assert_equal(actual_csv, desired_csv)

    def test_multiple_csvread(self):
        test_csv = './test.csv'
        actual_csv = CSVData([test_csv, test_csv]).read()
        desired_csv = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]])]
        assert_equal(actual_csv, desired_csv)

if __name__ == "__main__":
    run_module_suite()