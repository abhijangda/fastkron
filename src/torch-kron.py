import torch

def initmat(m, n):
    return torch.randint(0, 5, (m,n)) #.cuda()

def baseline(input, kronmats):
    outputKron = kronmats[0]
    for m in kronmats[1:]:
        outputKron = torch.kron(outputKron, m)
    return torch.matmul(input, outputKron)

def matmulkron(input, kronmats):
    output = input
    shape = input.shape
    kronmats.reverse()
    for k in kronmats:
        newinput = output.reshape(shape[0] * (shape[1]//k.shape[0]), k.shape[0])
        output = torch.matmul(newinput, k)
        output = output.view(shape[0], (shape[1]//k.shape[0]), k.shape[0])
        output = output.transpose(1,2)
    return output.reshape(shape)


npoints = 3
twoPower = 32
input = initmat(npoints, twoPower*twoPower)
kronmats = [initmat(twoPower,twoPower)]*2 #[initmat(twoPower,twoPower), initmat(twoPower,twoPower)]
# print (input)
# print(kronmats[0])
# print(kronmats[1])
b = baseline(input, kronmats)

o = matmulkron(input, kronmats)

print ((b == o))
print ((b == o).all())
