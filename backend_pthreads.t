backend = {}

C = terralib.includecstring([[
#include <pthreads.h>
#include <stdio.h>
#include <stdlib.h>
]])



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


struct VECDATA {
  gradE_d : &float,
  lam : int,
  uk_d : &float,
  input_d : &float,
  tid : int,
  total_length : int
}
terra launchPreparation(width : int, height : int, input_d : &float, gradE_d : &float, ukp1_d : &float, uk_d : &float, lam : float)
  
  return {data_t1, data_t2}
end
backend.launchPreparation = launchPreparation


terra launchKernelGrad(launch : &opaque, gradE_d : &&float, lam : float, uk_d : &&float, input_d : &&float, kernel : {&terralib.CUDAParams, &float, float, &float, &float} -> uint32)
  var data_t1 : VECDATA
  var data_t2 : VECDATA

  data_t1.gradE_d = gradE_d
  data_t1.lam = lam
  data_t1.uk_d = uk_d
  data_t1.input_d = input_d
  data_t1.tid = 0
  data_t1.total_length = width*height

  data_t1.gradE_d = gradE_d
  data_t1.lam = lam
  data_t1.uk_d = uk_d
  data_t1.input_d = input_d
  data_t1.tid = 0
  data_t1.total_length = width*height

  var t1 : C.pthread_t
  var t2 : C.pthread_t

  var data_t1 = launch[0]
  var data_t2 = launch[1]

  C.pthread_create(&t1, nil, kernel, &data_t1)
  C.pthread_create(&t2, nil, kernel, &data_t2)
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
    for y = 0,width do
      for x = 0,height do
        var idx : backend.Index
        if idx:initAndTestIfValid(width, height, x, y) then
            func(idx, args)
        end
      end
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


