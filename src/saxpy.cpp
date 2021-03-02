

// suppose you need to compute Kronecker Product of A and B, i.e., C = A x B then.
 // C[0: len(B)]  = saxpy(A[0][0], B).
// C[len(B) : len(B) * 2] = saxply(A[0][1], B).
// and so on
// 8:38
// do you see it?
// 8:38
// treat B as a vector of length N * M.

// This function multiplies the vector x by the scalar α and adds it to the vector y
// overwriting the latest vector with the result. Hence, the performed operation is
// y [ j ] = α × x [ k ] + y [ j ] for i = 1 , … , n , k = 1 + ( i - 1 ) *  incx and j = 1 + ( i - 1 ) *  incy .
// Notice that the last two equations reflect 1-based indexing used for compatibility with Fortran.
// cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
//                            const float           *alpha,
//                            const float           *x, int incx,
//                            float                 *y, int incy);
