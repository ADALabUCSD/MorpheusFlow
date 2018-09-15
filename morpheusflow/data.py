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


import abc
import pandas as pd
from scipy.io import mmread


class BaseData(abc.ABC):
    '''
    Base data class that can be passed into dataset API.
    '''
    def __init__(self, input):
        self._input = input

    @abc.abstractmethod
    def read(self):
        return "Base read not implemented"

    @property
    @abc.abstractmethod
    def sparse(self):
        return "Base sparse not implemented"

    @property
    @abc.abstractmethod
    def shape(self):
        return "Base shape not implemented"


class NumPyData(BaseData):
    '''
    NumPy array or matrix.
    Input can be either a NumPy array or a list of NumPy array.
    '''
    def read(self):
        return self._input

    @property
    def sparse(self):
        return False

    @property
    def shape(self):
        if isinstance(self._input, (list, tuple)):
            return [inp.shape for inp in self._input]
        else:
            return self._input.shape


class SparseData(BaseData):
    '''
    SciPy sparse matrix
    Input can be either a SciPy sparse matrix or a list of sparse matrices.
    '''
    def read(self):
        return self._input

    @property
    def sparse(self):
        return True

    @property
    def shape(self):
        if isinstance(self._input, (list, tuple)):
            return [inp.shape for inp in self._input]
        else:
            return self._input.shape


class CSVData(BaseData):
    '''
    CSV file
    Input can be either a csv file name or a list of csv file names.
    '''
    def __init__(self, input):
        super(CSVData, self).__init__(input)
        self._read = None

    def read(self):
        if self._read:
            return self._read
        if isinstance(self._input, (list, tuple)):
            self._read = [pd.read_csv(inp, sep=',').values for inp in self._input]
            self._shape = [r.shape for r in self._read]
            return self._read
        else:
            self._read = pd.read_csv(self._input, sep=',').values
            self._shape = self._read.shape
            return self._read

    @property
    def sparse(self):
        return False

    @property
    def shape(self):
        if self._shape:
            return self._shape
        else:
            print("Warning! Getting a shape of unread csv file will trigger the read!")
            self.read()
            return self._shape


class MatrixMarketData(BaseData):
    '''
    Matrix market file
    Input can be either a matrix market file name or a list of matrix market file names.
    '''
    def __init__(self, input):
        super(MatrixMarketData, self).__init__(input)
        self._read = None

    def read(self):
        if self._read:
            return self._read
        if isinstance(self._input, (list, tuple)):
            self._read = [mmread(inp,) for inp in self._input]
            self._shape = [r.shape for r in self._read]
            return self._read
        else:
            self._read = mmread(self._input,)
            self._shape = self._read.shape
            return self._read

    @property
    def sparse(self):
        return False

    @property
    def shape(self):
        if self._shape:
            return self._shape
        else:
            print("Warning! Getting a shape of unread matrix file will trigger the read!")
            self.read()
            return self._shape