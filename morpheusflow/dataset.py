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


import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from morpheusflow.utils import Converter as ct

class Dataset(object):
    def __init__(self, s, k, r, y):
        '''
        :param s: data object (entity table)
        :param k: data object (key map)
        :param r: data object (attribute table)
        :param y: data object (labels)
        '''
        self._k = [tf.data.Dataset.from_tensor_slices(ki) for ki in k.read()]
        self._y = tf.data.Dataset.from_tensor_slices(y.read())
        self._r = r.read()
        self._sparse = any(map(sp.issparse, self._r)) or (s is not None and s.sparse)

        if s is None:
            self._s = None
            self._dataset = tf.data.Dataset.zip(tuple(self._k + [self._y]))
        else:
            s = ct.convert_sparse_matrix_to_sparse_tensor(s.read()) if s.sparse else s.read()
            self._s = tf.data.Dataset.from_tensor_slices(s)
            self._dataset = tf.data.Dataset.zip(tuple([self._s] + self._k + [self._y]))
        self.__start__()

    def __start__(self):
        '''
        Start session

        :return:
        '''
        self._iterator = self._dataset.make_initializable_iterator()
        self._next_element = self._iterator.get_next()
        self._sess = tf.Session()
        self._sess.run(self._iterator.initializer)

    def shuffle(self):
        '''
        Shuffle the dataset
        '''
        self._dataset = self._dataset.shuffle()
        self.__start__()
        return self

    def repeat(self, epochs):
        '''
        Repeat the dataset for epoches

        :param epochs: number of times the dataset will be repeated
        :return:
        '''
        self._dataset = self._dataset.repeat(epochs)
        self.__start__()
        return self

    def batch(self, batch_size):
        '''
        Batch read
        '''
        self._dataset = self._dataset.batch(batch_size)
        self.__start__()
        return self

    def prefetch(self, fetch_size):
        '''
        Prefetch elements from this dataset.
        '''
        self._dataset = self._dataset.prefetch(fetch_size)
        self.__start__()
        return self

    def get_next(self):
        '''
        :return: batched x and y
        '''
        import time
        n_cost = []
        start = time.time()
        next_element = self._sess.run(self._next_element)
        n_cost.append(time.time() - start)
        if self._sparse:
            if self._s is None:
                x = sp.hstack([r[next_element[i]] for i, r in enumerate(self._r)])
            else:
                start = time.time()
                s = ct.convert_sparse_tensor_to_csr(next_element[0])
                n_cost.append(time.time() - start)
                start = time.time()
                tmp = [r[next_element[i + 1]] for i, r in enumerate(self._r)]
                n_cost.append(time.time() - start)
                start = time.time()
                x = sp.hstack([s] + tmp)
                n_cost.append(time.time() - start)
                # x = sp.hstack([s] + [r[next_element[i + 1]] for i, r in enumerate(self._r)])
            start = time.time()
            ran = ct.convert_coo_to_sparse_value(x)
            n_cost.append(time.time() - start)
            return ran, next_element[-1], n_cost
            # return ct.convert_coo_to_sparse_value(x), next_element[-1], n_cost
        else:
            if self._s is None:
                x = np.hstack([r[next_element[i]] for i, r in enumerate(self._r)])
            else:
                s = np.mat(next_element[0])
                start = time.time()
                tmp = [r[next_element[i + 1]] for i, r in enumerate(self._r)]
                n_cost.append(time.time() - start)
                start = time.time()
                x = np.hstack([s] + tmp)
                n_cost.append(time.time() - start)
                # x = np.hstack([s] + [r[next_element[i + 1]] for i, r in enumerate(self._r)])
            return x, next_element[-1], n_cost

