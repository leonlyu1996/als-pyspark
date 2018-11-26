from collections import namedtuple
import numpy as np
from scipy.optimize import nnls
from scipy.linalg.blas import dspr, daxpy
from scipy.linalg import cholesky
from numpy.linalg.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import pinv
import copy


# class NormalEqationBase(namedtuple("NormalEquation", ["tri_k", "ata", "atb", "da", "k"])):
#
#     def __reduce__(self):
#         return NormalEquation, (self.tri_k, self.ata, self.atb, self.da, self.k)


class NormalEquation(object):

    def __init__(self, k):
        self.tri_k = k * (k + 1) / 2
        self.ata = np.zeros(self.tri_k)
        self.atb = np.zeros(k)
        self.da = np.zeros(k)
        self.k = k

    def copy(self, a):
        for i in range(self.k):
            self.da[i] = a[i]

    def add(self, a, b, c=1.0):
        assert c > 0
        assert a.shape[0] == self.k
        self.copy(a)

        print "before dspr----------\nata {}".format(self.ata)
        print "da-------da {}".format(self.da)

        # use ata as return?
        self.ata = dspr(n=self.k, alpha=c, x=self.da, incx=1, ap=self.ata, lower=1)

        print "after dspr ==========\nata {}".format(self.ata)
        if b != 0:
            print "before daxpy ---------\natb {}".format(self.atb)
            self.atb = daxpy(x=self.da, y=self.atb, n=self.k, a=b, incx=1, incy=1)
            print "after daxpy ==========\n atb {}".format(self.atb)

        return self

    def merge(self, other):

        assert other.k == self.k
        self.ata = \
            daxpy(other.ata, self.ata, n=self.ata.shape[0], a=1.0, incx=1, incy=1)

        self.atb =\
            daxpy(other.atb, self.atb, n=self.atb.shape[0], a=1.0, incx=1, incy=1)

        return self

    def reset(self):
        self.ata.fill(0.0)
        self.atb.fill(0.0)

    # def __reduce__(self):
    #     return NormalEquation, (self.tri_k, self.ata, self.atb, self.da, self.k)


class LeastSquaresNESolver:

    def __init__(self):
        pass

    def solve(self, ne, lamd):
        pass


class NNLSSolver(LeastSquaresNESolver):

    def __init__(self):
        LeastSquaresNESolver.__init__(self)
        self.__rank = -1
        self.__ata = None
        self.__initialized = False

    def initialize(self, rank):
        if not self.__initialized:
            self.__rank = rank
            self.__ata = np.zeros(shape=(rank, rank))
            self.__initialized = True

        else:
            assert self.__rank == rank

    def solve(self, ne, lamd):
        rank = ne.k
        self.initialize(rank)
        self.fill_ata(ne.ata, lamd)
        x = nnls(self.__ata, ne.atb)
        ne.reset()
        return x

    def fill_ata(self, tri_ata, lamd):

        pos = 0
        for i in range(self.__rank):

            for j in range(i, self.__rank):

                a = tri_ata[pos]
                self.__ata[i, j] = a
                self.__ata[j, i] = a
                pos += 1

            self.__ata[i, i] += lamd


class CholeskySolver(LeastSquaresNESolver):

    def __init__(self):
        LeastSquaresNESolver.__init__(self)
        # TODO ?
        self.__ata = None
        # self.__k = None

    def solve(self, ne, lamd):
        k = ne.k
        j = 2
        # for i in range(ne.tri_k):
        #     ne.ata[i] += lamd
        #     i += j
        #     j += 1

        self.fill_ata(ne.ata, lamd, k)
        try:

            print "before inv {}".format(self.__ata)
            # inverse_ata = inv(self.__ata)
            # inverse_ata = cholesky(self.__ata)
            c_factor, lower = cho_factor(self.__ata, lower=False)
            x = cho_solve((c_factor, lower), ne.atb)

            print "Use cholesky"
            print "after inv {}".format(self.__ata)
        except LinAlgError:
            print "2-th leading minor of the array may be not positive definite"
            # inverse_ata = pinv(self.__ata)
            # x = np.dot(inverse_ata, ne.atb)
            x = copy.copy(ne.atb)
            # exit(1)

        print ">>>>>>>>>>X<<<<<<<<<< {}".format(x)
        ne.reset()
        return x

    def fill_ata(self, tri_ata, lamd, rank):
        pos = 0
        self.__ata = np.zeros(shape=(rank, rank))
        for i in range(rank):

            for j in range(i, rank):
                a = tri_ata[pos]
                self.__ata[i, j] = a
                self.__ata[j, i] = a
                pos += 1

            self.__ata[i, i] += lamd

        print "tri_data {}\n lambda {}".format(tri_ata, lamd)
