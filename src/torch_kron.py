import torch
import gpytorch as gp

def initmat(m, n):
    return torch.randint(0, 5, (m,n))

def baseline(input, kronmats):
    kronmats = gp.lazy.KroneckerProductLazyTensor(*kronmats)        
    print(kronmats.shape)
    print(input.shape)
    return input@kronmats
    # outputKron = kronmats[0]
    # for m in kronmats[1:]:
    #     outputKron = torch.kron(outputKron, m)
    # return torch.matmul(input, outputKron)

def matmulkron(input, kronmats):
    shape = input.shape
    input = input.T
    output = input
    print(20, shape)
    for k in kronmats:
        newinput = output.reshape(k.shape[0], shape[0] * (shape[1]//k.shape[0]))
        output = torch.matmul(k.T, newinput)
        # print(output)
        print(24, output.shape, k.shape, newinput.shape)
        output = output.view(k.shape[0], (shape[1]//k.shape[0]), shape[0])
        print(25, output.shape)
        output = output.transpose(-3,-2)
        print(28, output.shape)
        output = output.reshape(*[], -1, shape[0])
        print(30, output.shape)
        print(output)
    return output.mT


if __name__ == "__main__":
    npoints = 2
    twoPower = 3
    d = 2
    input = initmat(npoints, twoPower**d)
    kronmats = []
    for s in range(d):
        kronmats += [initmat(twoPower,twoPower)]
    # print(kronmats[0])
    # print(kronmats[1])
    b = baseline(torch.clone(input), [torch.clone(k) for k in kronmats])

    o = matmulkron(torch.clone(input), kronmats)
    print('b.shape',b.shape)
    print (b)
    print('o.shape',o.shape)
    print (o)
    print((b==o).all())