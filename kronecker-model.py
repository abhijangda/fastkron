#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import numpy as np
import gpytorch as gp 
import torch.nn.functional as F

from matplotlib import pyplot as plt


# In[2]:


# model and data 

inputDim = 64   # takes variable 'x' 
outputDim = 1       # takes variable 'y'

npoints = 65536
x_train = np.zeros((npoints, inputDim)).astype(np.float32)
y_train = None #func(x_train)

# train adn prediciton 
def train_and_predict(model, x_train, y_train, verbose=False, trans=False, print_model=False):
    learningRate = 0.1
    epochs = 10

    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    model.cuda()
    for epoch in range(epochs):
        inputs = torch.from_numpy(x_train)
        # labels = torch.from_numpy(y_train)
        
        inputs = inputs.cuda()
        # labels = labels.cuda()

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs.T if trans else inputs)

        # get loss for the predicted output
        # loss = criterion(outputs, labels)

        # get gradients w.r.t to parameters
        # loss.backward()

    
        # update parameters
        # optimizer.step()
    
    # with torch.no_grad(): # we don't need gradients in the testing phase
    #     x_train = torch.from_numpy(x_train)
        
    #     predicted = model(x_train.T if trans else x_train).data.numpy()
# 
    # print("train mse: ", np.mean(y_train - predicted))

# In[6]:


# revised model and data 

from torch.nn.parameter import Parameter

class NewlinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, numFactors):
        super(NewlinearRegression, self).__init__()
        
        self.weights = []
        l = int(inputSize**(1./numFactors))
        print (l, numFactors)
        for i in range(numFactors):
            self.weights += [Parameter(torch.ones(l,l).cuda())]    
        
        self.l1_weight = gp.lazy.KroneckerProductLazyTensor(*self.weights)
                
        self.linear2 = torch.nn.Linear(inputSize, 1)
        self.inputSize = inputSize

    def count_params(self):
        return sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        print(x.shape)
        start.record()
        l1 = self.l1_weight@x
        end.record()
        torch.cuda.synchronize()
        exec_time = start.elapsed_time(end)
        bandwidth = 4 * 2 * (self.count_params() + x.numel() + l1.numel())/(exec_time/1e3)/1e9 
        flops = 2 * (self.inputSize * self.inputSize + x.shape[0] * x.shape[1] * x.shape[1])/(exec_time/1e3)/1e9
        print(len(self.weights), "Kronecker", exec_time, "= ms", "bandwidth = ", bandwidth, " GBPS", "elements = ", (self.count_params() + x.numel() + l1.numel()))
        
        # tmpT = torch.ones((inputDim, inputDim)).cuda()
        # start.record()
        # l1 = tmpT@x
        # end.record()
        # torch.cuda.synchronize()
        # exec_time = start.elapsed_time(end)
        # bandwidth = 4 * 2 * (inputDim*inputDim + x.numel() + l1.numel())/(exec_time/1e3)/1e9 
        # print("Matmul", exec_time, "= ms", "bandwidth = ", bandwidth, " GBPS", "elements = ", (inputDim*inputDim + x.numel() + l1.numel()))
        

        # l1_out = F.relu(l1)
        # out = self.linear2(l1.T)
        return l1 #out.squeeze()


for i in range(1, 2):
    model2 = NewlinearRegression(inputDim, outputDim, 2**i)
    train_and_predict(model2, x_train, y_train, True, trans=True, print_model=True)


