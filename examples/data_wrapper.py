import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
from morpheusflow.data import SparseData, NumPyData


def flights():
    r1 = mmread('./data/Flights/MLR1Sparse.txt', ).tocsr()
    r2 = mmread('./data/Flights/MLR2Sparse.txt', ).tocsr()
    r3 = mmread('./data/Flights/MLR3Sparse.txt', ).tocsr()

    k1 = np.genfromtxt('./data/Flights/MLFK1.csv', skip_header=True, dtype=int) - 1
    k2 = np.genfromtxt('./data/Flights/MLFK2.csv', skip_header=True, dtype=int) - 1
    k3 = np.genfromtxt('./data/Flights/MLFK3.csv', skip_header=True, dtype=int) - 1
    s = mmread('./data/Flights/MLSSparse.txt', )
    Y = np.matrix(np.genfromtxt('./data/Flights/MLY.csv', skip_header=True, dtype=int)).T

    return SparseData(s), SparseData([r1, r2, r3]), SparseData([k1, k2, k3]), NumPyData(Y), s.shape[0], s.shape[1] + r1.shape[1] + r2.shape[1] + r3.shape[1]

def expedia():
    r1 = mmread('./data/Expedia/MLR1Sparse.txt', ).tocsr()
    r2 = mmread('./data/Expedia/MLR2Sparse.txt', ).tocsr()

    k1 = np.genfromtxt('./data/Expedia/MLFK1.csv', skip_header=True, dtype=int) - 1
    k2 = np.genfromtxt('./data/Expedia/MLFK2.csv', skip_header=True, dtype=int) - 1
    s = mmread('./data/Expedia/MLSSparse.txt', )
    Y = np.matrix(np.genfromtxt('./data/Expedia/MLY.csv', skip_header=True, dtype=int)).T

    return SparseData(s), SparseData([r1, r2]), SparseData([k1, k2]), NumPyData(Y), s.shape[0], s.shape[1] + r1.shape[1] + r2.shape[1]

def yelp():
    r1 = mmread('./data/Yelp/MLR1Sparse.txt', ).tocsr()
    r2 = mmread('./data/Yelp/MLR2Sparse.txt', ).tocsr()

    k1 = np.genfromtxt('./data/Yelp/MLFK1.csv', skip_header=True, dtype=int) - 1
    k2 = np.genfromtxt('./data/Yelp/MLFK2.csv', skip_header=True, dtype=int) - 1
    Y = np.matrix(np.genfromtxt('./data/Yelp/MLY.csv', skip_header=True, dtype=int)).T

    return None, SparseData([r1, r2]), NumPyData([k1, k2]), NumPyData(Y), len(k1), r1.shape[1] + r2.shape[1]

def movie():
    r1 = mmread('./data/MovieLens1M/MLR1Sparse.txt', ).tocsr()
    r2 = mmread('./data/MovieLens1M/MLR2Sparse.txt', ).tocsr()

    k1 = np.genfromtxt('./data/MovieLens1M/MLFK1.csv', skip_header=True, dtype=int) - 1
    k2 = np.genfromtxt('./data/MovieLens1M/MLFK2.csv', skip_header=True, dtype=int) - 1
    Y = np.matrix(np.genfromtxt('./data/MovieLens1M/MLY.csv', skip_header=True, dtype=int)).T

    return None, SparseData([r1, r2]), SparseData([k1, k2]), NumPyData(Y), len(k1), r1.shape[1] + r2.shape[1]

def flights_materialized():
    r1 = mmread('./data/Flights/MLR1Sparse.txt', ).tocsr()
    r2 = mmread('./data/Flights/MLR2Sparse.txt', ).tocsr()
    r3 = mmread('./data/Flights/MLR3Sparse.txt', ).tocsr()

    k1 = np.genfromtxt('./data/Flights/MLFK1.csv', skip_header=True, dtype=int) - 1
    k2 = np.genfromtxt('./data/Flights/MLFK2.csv', skip_header=True, dtype=int) - 1
    k3 = np.genfromtxt('./data/Flights/MLFK3.csv', skip_header=True, dtype=int) - 1
    s = mmread('./data/Flights/MLSSparse.txt', )
    Y = np.matrix(np.genfromtxt('./data/Flights/MLY.csv', skip_header=True, dtype=int)).T

    return sp.hstack((s, r1.tocsr()[k1], r2.tocsr()[k2], r3.tocsr()[k3])), NumPyData(Y), s.shape[1] + r1.shape[1] + r2.shape[1] + r3.shape[1]

def expedia_materialized():
    r1 = mmread('./data/Expedia/MLR1Sparse.txt', ).tocsr()
    r2 = mmread('./data/Expedia/MLR2Sparse.txt', ).tocsr()

    k1 = np.genfromtxt('./data/Expedia/MLFK1.csv', skip_header=True, dtype=int) - 1
    k2 = np.genfromtxt('./data/Expedia/MLFK2.csv', skip_header=True, dtype=int) - 1
    s = mmread('./data/Expedia/MLSSparse.txt', )
    Y = np.matrix(np.genfromtxt('./data/Expedia/MLY.csv', skip_header=True, dtype=int)).T

    return sp.hstack((s, r1.tocsr()[k1], r2.tocsr()[k2])), NumPyData(Y), s.shape[1] + r1.shape[1] + r2.shape[1]

def yelp_materialized():
    r1 = mmread('./data/Yelp/MLR1Sparse.txt', ).tocsr()
    r2 = mmread('./data/Yelp/MLR2Sparse.txt', ).tocsr()

    k1 = np.genfromtxt('./data/Yelp/MLFK1.csv', skip_header=True, dtype=int) - 1
    k2 = np.genfromtxt('./data/Yelp/MLFK2.csv', skip_header=True, dtype=int) - 1
    Y = np.matrix(np.genfromtxt('./data/Yelp/MLY.csv', skip_header=True, dtype=int)).T

    return sp.hstack((r1.tocsr()[k1], r2.tocsr()[k2])), NumPyData(Y), r1.shape[1] + r2.shape[1]

def movie_materialized():
    r1 = mmread('./data/MovieLens1M/MLR1Sparse.txt', ).tocsr()
    r2 = mmread('./data/MovieLens1M/MLR2Sparse.txt', ).tocsr()

    k1 = np.genfromtxt('./data/MovieLens1M/MLFK1.csv', skip_header=True, dtype=int) - 1
    k2 = np.genfromtxt('./data/MovieLens1M/MLFK2.csv', skip_header=True, dtype=int) - 1
    Y = np.matrix(np.genfromtxt('./data/MovieLens1M/MLY.csv', skip_header=True, dtype=int)).T

    return sp.hstack((r1.tocsr()[k1], r2.tocsr()[k2])), NumPyData(Y), r1.shape[1] + r2.shape[1]