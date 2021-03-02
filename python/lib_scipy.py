
import scipy
from scipy import sparse
import time
import numpy as np

def test_scipy_kronecker(A_dim1,A_dim2,B_dim1,B_dim2):
    A = np.random.rand(A_dim1,A_dim2)
    B = np.random.rand(B_dim1,B_dim2)
    A_sp = sparse.csr_matrix(np.array([[0, 2], [5, 0]]))
    B_sp = sparse.csr_matrix(np.array([[1, 2], [3, 4]]))
    s = time.time()
    sparse.kron(A_sp, B_sp).toarray()
    e = time.time()
    print("Sparse scipy",e-s)
    # s = time.time()
    # scipy.linalg.kron(A, B)
    # e = time.time()
    # print("Dense scipy",e-s)

if __name__=="__main__":
    test_scipy_kronecker(100,100,100,100)
