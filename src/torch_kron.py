import torch

def initmat(m, n):
    return torch.randn(m,n)

def baseline(input, kronmats):
    outputKron = kronmats[0]
    for m in kronmats[1:]:
        outputKron = torch.kron(outputKron, m)
    return torch.matmul(input, outputKron)

def matmulkron(input, kronmats):
    output = input
    shape = input.shape
    
    for k in reversed(kronmats):
        newinput = output.reshape(shape[0] * (shape[1]//k.shape[0]), k.shape[0])
        output = torch.matmul(newinput, k)
        output = output.view(shape[0], (shape[1]//k.shape[0]), k.shape[0])
        output = output.transpose(1,2)
    return output.reshape(shape)


if __name__ == "__main__":
    npoints = 2
    twoPower = 8
    input = initmat(npoints, twoPower**3)
    kronmats = []
    for s in range(3):
        kronmats += [initmat(twoPower,twoPower)]
    # print(kronmats[0])
    # print(kronmats[1])
    b = baseline(input, kronmats)

    o = matmulkron(input, kronmats)

    print ((b == o))
    print ((b == o).all())