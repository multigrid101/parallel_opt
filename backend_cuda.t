backend = {}

C = terralib.includecstring([[
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
]])

-- not sure if I need the next two.....
C.vprintf = terralib.externfunction("cudart:vprintf", {&int8,&int8} -> int)
C.sync = terralib.externfunction("cudaThreadSynchronize", {} -> int)

if not terralib.cudacompile then
  print("CUDA not enabled!")
end


-- some shortcuts for better readability
local tid_x = cudalib.nvvm_read_ptx_sreg_tid_x
local tid_y = cudalib.nvvm_read_ptx_sreg_tid_y
local blockDim_x = cudalib.nvvm_read_ptx_sreg_ntid_x
local blockDim_y = cudalib.nvvm_read_ptx_sreg_ntid_y
local gridDim_x = cudalib.nvvm_read_ptx_sreg_nctaid_x
local gridDim_y = cudalib.nvvm_read_ptx_sreg_nctaid_y
local blockIdx_x = cudalib.nvvm_read_ptx_sreg_ctaid_x
local blockIdx_y = cudalib.nvvm_read_ptx_sreg_ctaid_y



struct Index { 
               x : int,
               y : int 
             }
terra Index:initAndTestIfValid(width : int, height : int)
  self.x = blockIdx_x()*blockDim_x() + tid_x()
  self.y = blockIdx_y()*blockDim_y() + tid_y()

  var isvalid : bool = (self.x>=0) and (self.x<=width-1) and (self.y>=0) and (self.y<=height-1)
  return isvalid
end
backend.Index = Index


terra allocation(input_d : &&float, gradE_d : &&float, ukp1_d : &&float, uk_d : &&float, numpixels : int)
  C.cudaMalloc([&&opaque](input_d), numpixels * sizeof(float))
  C.cudaMalloc([&&opaque](gradE_d), numpixels * sizeof(float))
  C.cudaMalloc([&&opaque](ukp1_d), numpixels * sizeof(float))
  C.cudaMalloc([&&opaque](uk_d), numpixels * sizeof(float))
end
backend.allocation = allocation


terra initialization(input_d : &&float, input_h : &&float, uk_d : &&float, output_h : &&float, numpixels : int)
  C.cudaMemcpy(@input_d, @input_h, numpixels * sizeof(float), C.cudaMemcpyHostToDevice)
  C.cudaMemcpy(@uk_d, @output_h, numpixels * sizeof(float), C.cudaMemcpyHostToDevice)
end
backend.initialization = initialization

terra launchPreparation(width : int, height : int)
  var blockSize = 32
  var gridSizeX = width/blockSize +1
  var gridSizeY = height/blockSize +1
  var launch = terralib.CUDAParams {gridSizeX,gridSizeY,1, blockSize,blockSize,1, 0, nil}

  return launch
end
backend.launchPreparation = launchPreparation


terra launchKernelGrad(launch : terralib.CUDAParams, gradE_d : &&float, lam : float, uk_d : &&float, input_d : &&float, kernel : {&terralib.CUDAParams, &float, float, &float, &float} -> uint32)
  kernel(&launch, @gradE_d, lam, @uk_d, @input_d)
end
backend.launchKernelGrad = launchKernelGrad


terra launchKernelUkp1(launch : terralib.CUDAParams, ukp1_d : &&float, tau : float, uk_d : &&float, gradE_d : &&float, kernel : {&terralib.CUDAParams, &float, float, &float, &float} -> uint32)
  kernel(&launch, @ukp1_d, tau, @uk_d, @gradE_d)
end
backend.launchKernelUkp1 = launchKernelUkp1


terra retrieval(output_h : &&float, uk_d : &&float, numpixels : int)
  C.cudaMemcpy(@output_h, @uk_d, numpixels * sizeof(float), C.cudaMemcpyDeviceToHost)
end
backend.retrieval = retrieval


function makeloop(func, args)
  return quote
    var idx : backend.Index
    if idx:initAndTestIfValid(width, height) then
        func(idx, args)
    end
  end
end
backend.makeloop = makeloop


function makeKernelList(rawlist)
  local kernels = terralib.cudacompile({kernelGrad = rawlist[1], kernelUkp1 = rawlist[2]})
  return kernels
end
backend.makeKernelList = makeKernelList

return backend


