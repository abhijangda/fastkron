
import numpy as np
import time
def kronecker(a,b):
    c = np.kron(a,b)


def test_numpy_kronecker(A_dim1,A_dim2, B_dim1, B_dim2):
    A = np.random.rand(A_dim1,A_dim2)
    B = np.random.rand(B_dim1,B_dim2)
    s = time.time()
    kronecker(A,B)
    e = time.time()
    print("Time taken numpy",e-s)

if __name__=="__main__":
    test_numpy_kronecker(100,100,100,100)
    # np.rand.random(100,100)
    # np.rand.random(100,100)
