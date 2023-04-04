import numpy as np
import fastmat as fm


# Source:- https://gist.github.com/ahwillia/f65bc70cb30206d4eadec857b98c4065
def unfold(tens, mode, dims):
    """
    Unfolds tensor into matrix.
    Parameters
    ----------
    tens : ndarray, tensor with shape == dims
    mode : int, which axis to move to the front
    dims : list, holds tensor shape
    Returns
    -------
    matrix : ndarray, shape (dims[mode], prod(dims[/mode]))
    """
    if mode == 0:
        return tens.reshape(dims[0], -1)
    else:
        return np.moveaxis(tens, mode, 0).reshape(dims[mode], -1)


# Source:- https://gist.github.com/ahwillia/f65bc70cb30206d4eadec857b98c4065
def refold(vec, mode, dims):
    """
    Refolds vector into tensor.
    Parameters
    ----------
    vec : ndarray, tensor with len == prod(dims)
    mode : int, which axis was unfolded along.
    dims : list, holds tensor shape
    Returns
    -------
    tens : ndarray, tensor with shape == dims
    """
    if mode == 0:
        return vec.reshape(dims)
    else:
        # Reshape and then move dims[mode] back to its
        # appropriate spot (undoing the `unfold` operation).
        tens = vec.reshape(
            [dims[mode]] +
            [d for m, d in enumerate(dims) if m != mode]
        )
        return np.moveaxis(tens, 0, mode)


class TestVectorShape(object):

    def __init__(self, shape, dtype):

        self._shape = shape
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def process_vector(self, vector, shape=None):
        shape = self.shape if shape is None else shape
        if len(vector.shape) == 1:
            assert len(vector) == shape[-1], str(len(vector)) + ' !=  ' + str(shape[-1])
            vector = vector.reshape(-1, 1)
        elif len(vector.shape) == 2:
            assert vector.shape[1] == 1, "Vector shape: " + str(vector.shape)
        else:
            raise ValueError('Expected vector ... got incompatible shape, i.e., ' + str(vector.shape))
        return np.array(vector)

    def matvec(self, x):
        raise NotImplementedError

    def __matmul__(self, x):
        assert len(x.shape) <= 1 or (len(x.shape) == 2 and x.shape[1])  # Not tested for x as a matrix
        return self.matvec(x)


class KroneckerToeplitzMatVec(TestVectorShape):

    def __init__(self, kmm, shape, dtype):
        """
        :param kmm: kernel grid matrix
        :param shape: shape of the kernel grid matrix
        :param dtype:
        """
        super().__init__(shape=shape, dtype=dtype)
        self.kmm = kmm
        self.is_one_dim = len(kmm) == 1
        self.kmm_shape = (np.prod([item.shape[0] for item in kmm]), np.prod([item.shape[0] for item in kmm]))

    def kmm_matvec(self, vector):

        """
        :param vector:
        :return:
        """

        # processing input vector
        vector = super().process_vector(vector, shape=self.kmm_shape)

        if self.is_one_dim:
            return (self.kmm[0] * vector).real.squeeze().reshape(-1, 1)

        dims = [A.shape[0] for A in self.kmm]
        vt = vector.reshape(dims)
        for i in range(len(self.kmm)):
            vt = refold(self.kmm[i] * unfold(vt, i, dims), i, dims)
        return vt.ravel().real.reshape(-1, 1)

    def kmm_matmat(self, matrix):

        """
        :param matrix: (n, k)
        :return:
        """

        # processing input vector
        assert len(matrix.shape) == 2, "Not 2 dimensional ..."

        result_matrix = []
        for i in range(matrix.shape[1]):
            result_matrix += self.kmm_matvec(matrix[:, i]),
        return np.hstack(result_matrix)


class KissGpLinearOperator(KroneckerToeplitzMatVec):

    def __init__(self, W, kmm, sigma, dtype):

        """
        Linear operator for KissGP that provides interface to MVM
            - in O(n + d * (m1 * ... * md)  * log (m1 * ... * md))
            - for n data-points, d dimensions, mi inducing points in ith dimension.

        :param W: sparse data matrix of shape (n, (m1 * ... * md)).
        :param kmm: Kernel matrix over grid points for each dimension as a list [(m, m), ..., (m, m)].
        :param sigma: Noise value.
        :param dtype: data_type of kernel matrix
        """

        super().__init__(kmm=kmm, shape=(W.shape[0], W.shape[0]), dtype=dtype)

        self.W = fm.Sparse(W)
        self.WT = fm.Sparse(W.T.tocsr())
        self.sigma = sigma
        self.kmm_shape = (np.prod([item.shape[0] for item in kmm]), np.prod([item.shape[0] for item in kmm]))

    def matvec(self, vector):
        """
        :param vector: the probe vector of shape (n, 1)
        :return: [W * K * W^T + sigma^2 I_n] * vector of shape (n, 1)
        """

        vector = super().process_vector(vector)
        vector = (self.W * (self.kmm_matvec(self.WT * vector)) + self.sigma ** 2 * vector)
        return vector.real

    def matmat(self, matrix):
        """
        :param matrix: the probe matrix of size (n, k)
        :return: [W * K * W^T + sigma^2 I_n] * matrix of size (n, k)
        """

        assert len(matrix.shape) == 2
        return (self.W * self.kmm_matmat(self.WT * matrix) + self.sigma ** 2 * matrix).real
