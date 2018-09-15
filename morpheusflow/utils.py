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
import tensorflow as tf
import scipy.sparse as sp

class Converter(object):
    @staticmethod
    def convert_sparse_tensor_to_csr(tensor):
        indices = np.array(tensor.indices)
        if len(tensor.dense_shape) == 1:
            return sp.csr_matrix((tensor.values, (np.zeros(tensor.dense_shape[0]), indices[:, 0])), shape=(1, tensor.dense_shape[0]))
        else:
            return sp.csr_matrix((tensor.values, (indices[:,0], indices[:, 1])), shape=tensor.dense_shape)

    @staticmethod
    def convert_sparse_tensor_to_coo(tensor):
        indices = np.array(tensor.indices)
        return sp.coo_matrix((tensor.values, (indices[:,0], indices[:, 1])), shape=tensor.dense_shape)

    @staticmethod
    def convert_sparse_matrix_to_sparse_tensor(sparse):
        s_format = sparse.getformat()

        if s_format == "csr":
            return Converter.convert_csr_to_sparse_tensor(sparse)
        elif s_format == "coo":
            return Converter.convert_coo_to_sparse_tensor(sparse)
        elif s_format == "csc":
            return Converter.convert_coo_to_sparse_tensor(sparse.to_coo())
        else:
            raise Exception("Failed to convert sparse matrix to sparse tensor.")

    @staticmethod
    def convert_coo_to_sparse_tensor(coo):
        indices = np.mat([coo.row, coo.col]).transpose()

        return tf.sparse_reorder(tf.SparseTensor(indices, coo.data, coo.shape))

    @staticmethod
    def convert_csr_to_sparse_tensor(csr):
        indices = np.mat(csr.nonzero()).transpose()
        return tf.SparseTensor(indices, csr.data, csr.shape)

    @staticmethod
    def convert_coo_to_sparse_value(coo):
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensorValue(indices, coo.data, coo.shape)

    @staticmethod
    def convert_csr_to_sparse_value(csr):
        indices = np.mat(csr.nonzero()).transpose()
        return tf.SparseTensorValue(indices, csr.data, csr.shape)