#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import numpy as np
import gpytorch as gp 
import torch.nn.functional as F

from matplotlib import pyplot as plt
import math

# In[2]:

use_torch_profiler = True
epochs = 10

# model and data 

def do(twoPowerL, npoints, d):
    outputDim = 1       # takes variable 'y'
    inputDim = twoPowerL ** d
    x_train = np.zeros((npoints, inputDim)).astype(np.float32)
    y_train = None #func(x_train)

    # train adn prediciton 
    def train_and_predict(model, x_train, y_train, verbose=False, trans=False, print_model=False):
        learningRate = 0.1

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
            l = int(round(inputSize**(1./numFactors)))

            for i in range(numFactors):
                self.weights += [Parameter(torch.ones(l,l).cuda())]    
            
            self.l1_weight = gp.lazy.KroneckerProductLazyTensor(*self.weights)
                    
            self.linear2 = torch.nn.Linear(inputSize, 1)
            self.inputSize = inputSize
            self.all_cuda_times = []
            self.all_cublas_times = []
            self.all_at_times = []

        def count_params(self):
            return sum([p.numel() for p in self.parameters()])

        def forward(self, x):
            if use_torch_profiler:
                with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
                ) as p:
                    print(self.l1_weight.shape, x.shape)
                    l1 = self.l1_weight@x
                    torch.cuda.synchronize()
            
                cuda_time = 0
                cublas_time = 0
                at_time = 0
                for event in p.events():
                    if "sgemm" in event.name or "gemmSN" in event.name:
                        cublas_time += event.cuda_time
                    elif "at::native::elementwise_kernel" in event.name:
                        at_time += event.cuda_time

                    if event.device_type == torch._C._autograd.DeviceType.CUDA:
                        cuda_time += event.cuda_time
                exec_time = cuda_time #start.elapsed_time(end)
                self.all_cuda_times.append(cuda_time)
                self.all_cublas_times.append(cublas_time)
                self.all_at_times.append(at_time)
            else:
                l1 = self.l1_weight@x
                torch.cuda.synchronize()
            return

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


    # for i in range(2, 3):
    model = NewlinearRegression(inputDim, outputDim, d)
    train_and_predict(model, x_train, y_train, True, trans=True,print_model=True)
    all_cuda_times = model.all_cuda_times
    all_cublas_times = model.all_cublas_times
    all_at_times = model.all_at_times
    del model
    torch.cuda.empty_cache()
    return all_cublas_times, all_at_times, all_cuda_times

maxD = {2:22, 4:11, 8:7, 16: 5, 32: 4, 64 : 3} #128:2

cases = [{"npoints": 100, "2^l": j, "d": i} for j in maxD for i in range(2 if j > 4 else 4, maxD[j]+1)] 
#  [       {"npoints": 100, "2^l": 32, "d": 2},
#         {"npoints": 10, "2^l": 32, "d": 2},
#         {"npoints": 1, "2^l": 32, "d": 2},
#         # {"npoints": 65536, "2^l^d": 1024, "d": 2},
        
#         # {"npoints": 100, "2^l^d": 256, "d": 2},
#         {"npoints": 100, "2^l": 4, "d": 5},
#         {"npoints": 10, "2^l": 4, "d": 5},
#         {"npoints": 1, "2^l": 4, "d": 5},
#         {"npoints": 100, "2^l": 2, "d": 10},
#         {"npoints": 10, "2^l": 2, "d": 10},
#         {"npoints": 1, "2^l": 2, "d": 10},

#         {"npoints": 100, "2^l": 4, "d": 4},
#         {"npoints": 100, "2^l": 4, "d": 5},
#         {"npoints": 100, "2^l": 4, "d": 6},
#         {"npoints": 100, "2^l": 4, "d": 7},
#         {"npoints": 100, "2^l": 4, "d": 8},
#         {"npoints": 100, "2^l": 4, "d": 9},
#         {"npoints": 100, "2^l": 4, "d": 10},

#         {"npoints": 100, "2^l": 8, "d": 5},
        
#         {"npoints": 10, "2^l": 4, "d": 4},
#         {"npoints": 1, "2^l": 4, "d": 4},

#         {"npoints": 100, "2^l": 4, "d": 6},
#         {"npoints": 1, "2^l": 4, "d": 6},
#         ]

import subprocess
import sys 

case_times = {}
for case in cases:
        if True:
            (cublas_times, at_times, cuda_times) = do(case["2^l"], case["npoints"], case["d"])
            case["PyTorchTime"] = sum(cuda_times[1:])/len(cuda_times[1:])
            case["cuBLASTime"] = sum(cublas_times[1:])/len(cublas_times[1:])
            case["atTime"] = sum(at_times[1:])/len(at_times[1:])
            bandwidth = 4 * 2 * (case["npoints"] * (case["2^l"] ** case["d"]))/(case["PyTorchTime"]/1e6)/1e9
            case["PyTorchBandwidth"] = bandwidth
        
            # case["PyTorchTime"] = -1
        twoPowerL = case["2^l"]

        (s, o) = subprocess.getstatusoutput("./kron-eval %d %d %d"%(case["npoints"], case["d"], twoPowerL))
        if s != 0:
            print(o)
            case["CUDATime"] = -1
            case["Speedup"] = -1
        else:
            kront = float(o[o.find("elapsedtime ") + len("elapsedtime"):o.find("milliseconds")].strip()) * 1000 #Convert ms to us
            case["CUDATime"] = kront
            case["Speedup"] = case["PyTorchTime"]/case["CUDATime"]
            print(case)

row_format = "{:>20}" * 8
print(row_format.format("Batch-Size", "d", "2^l", "PyTorchTime(us)", "cuBLASTime(us)", "atTime(us)", "CUDATime(us)", "Speedup"))
for case in cases:
    twoPowerL = case["2^l"]
    print(row_format.format(case["npoints"], case["d"],twoPowerL, "%.3f"%case["PyTorchTime"], "%.3f"%case["cuBLASTime"], "%.3f"%case["atTime"], "%.3f"%case["CUDATime"], "%.3f"%case["Speedup"]))
