
import torch,time

def test_torch_kronecker(A_dim1,A_dim2,B_dim1,B_dim2):
    A = torch.rand((A_dim1,A_dim2)).to('cuda')
    B = torch.rand((B_dim1,B_dim2)).to('cuda')
    s_t = time.time()
    torch.kron(A, B)
    e_t = time.time()
    print("Torch time",e_t - s_t)

if __name__=="__main__":
    test_torch_kronecker(100,100,100,100)
