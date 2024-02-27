import torch

def initmat(m, n):
    return torch.randint(0, 10, (m,n)) #randn(m,n)

def baseline(input, kronmats):
    outputKron = kronmats[0]
    for m in kronmats[1:]:
        outputKron = torch.kron(outputKron, m)
    return torch.matmul(input, outputKron)

def matmulkron(input, kronmats):
    output = input
    shape = input.shape
    
    for i,k in enumerate(reversed(kronmats)):
        newinput = output.reshape(shape[0] * (shape[1]//k.shape[0]), k.shape[0])
        output = torch.matmul(newinput, k)
        output = output.view(shape[0], (shape[1]//k.shape[0]), k.shape[0])
        output = output.transpose(1,2)
        # if i == 1:
        #print(output.shape)
    return output.reshape(shape)

def contraction(input, kronmats):
    output = input
    shape = input.shape
    n = len(kronmats)
    output = output.reshape([shape[0],] + [kronmats[0].shape[0] for i in range(len(kronmats))])

    for i,k in enumerate(reversed(kronmats)):
        print(i, output.shape)
        output = torch.tensordot(output, k, dims=([n],[0]))
        for j in range(n, 1,-1):
            output = output.transpose(j, j-1)
    return output.reshape(shape)

def fackler2019toms(input, kronmats):
    #https://dl-acm-org.silk.library.umass.edu/doi/pdf/10.1145/3291041
    output = input
    shape = input.shape
    n = len(kronmats)
    output = output.reshape([kronmats[0].shape[0] for i in range(len(kronmats))] + [shape[0],])
    kronrows = kronmats[0].shape[0]
    for i,k in enumerate(reversed(kronmats)):
        print(i, output.shape)
        output = output.mT @ k.T
        output = output.reshape([kronmats[0].shape[0] for j in range(len(kronmats) - i - 1)] + [shape[0],] + [kronmats[0].shape[0] for j in range(i+1)])

    return output.reshape(shape)

#[5,5], [4,4], [3,3]
#[N, 3, 4, 5] x [5,5] = [N,3,4,5]
#[N,3,5,4] x [4,4] = [N,3,5,4]
#[N,4,5,3]x[3,3] = [N,4,5,3]

if __name__ == "__main__":
    npoints = 3
    twoPower = 4
    input = initmat(npoints, twoPower**npoints)
    kronmats = []
    for s in range(npoints):
        kronmats += [initmat(twoPower,twoPower)]
    # print(kronmats[0])
    # print(kronmats[1])
    b = baseline(input, kronmats)

    o = matmulkron(input, kronmats)
    # o = fackler2019toms(input, kronmats)
    # o = contraction(input, kronmats)

    print ((b == o))
    print ((b == o).all())