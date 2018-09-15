import abc
import pandas as pd


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


class NumPyData(BaseData):
    def read(self):
        return self._input

    @property
    def sparse(self):
        return False


class SparseData(BaseData):
    def read(self):
        return self._input

    @property
    def sparse(self):
        return True


class CSVData(BaseData):
    def read(self):
        if isinstance(self._input, (list, tuple)):
            return [pd.read_csv(inp, sep=',').values for inp in self._input]
        else:
            df = pd.read_csv(self._input, sep=',')
            return df.values

    @property
    def sparse(self):
        return False