
import scipy

def test_scipy_kronecker(A_dim1,A_dim2,B_dim1,B_dim2):
    from scipy import sparse

A = sparse.csr_matrix(np.array([[0, 2], [5, 0]]))

B = sparse.csr_matrix(np.array([[1, 2], [3, 4]]))

sparse.kron(A, B).toarray()


 scipy.linalg.kron(a, b)


if __name__=="__main__":
    test_scipy_kronecker(A_dim1,A_dim2,B_dim1,B_dim2)
