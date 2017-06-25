backend = {}

C = terralib.includecstring([[
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
]])




struct Index { 
               x : int,
               y : int 
             }
terra Index:initAndTestIfValid(width : int, height : int, x:int, y:int)
  self.x = x
  self.y = y

  var isvalid : bool = (self.x>=0) and (self.x<=width-1) and (self.y>=0) and (self.y<=height-1)
  return isvalid
end
backend.Index = Index


terra allocation(input_d : &&float, gradE_d : &&float, ukp1_d : &&float, uk_d : &&float, numpixels : int)
  @input_d = [&float](C.malloc(numpixels * sizeof(float)))
  @gradE_d = [&float](C.malloc(numpixels * sizeof(float)))
  @ukp1_d = [&float](C.malloc(numpixels * sizeof(float)))
  @uk_d = [&float](C.malloc(numpixels * sizeof(float)))
end
backend.allocation = allocation


terra initialization(input_d : &&float, input_h : &&float, uk_d : &&float, output_h : &&float, numpixels : int)
  C.memcpy(@input_d, @input_h, numpixels * sizeof(float))
  C.memcpy(@uk_d, @output_h, numpixels * sizeof(float))
end
backend.initialization = initialization

terra launchPreparation(width : int, height : int)
  return true
end
backend.launchPreparation = launchPreparation


terra launchKernelGrad(launch : bool, gradE_d : &&float, lam : float, uk_d : &&float, input_d : &&float, kernel : {&float, float, &float, &float} -> {})
  kernel(@gradE_d, lam, @uk_d, @input_d)
end
backend.launchKernelGrad = launchKernelGrad


terra launchKernelUkp1(launch : bool, ukp1_d : &&float, tau : float, uk_d : &&float, gradE_d : &&float, kernel : {&float, float, &float, &float} -> {})
  kernel(@ukp1_d, tau, @uk_d, @gradE_d)
end
backend.launchKernelUkp1 = launchKernelUkp1


terra retrieval(output_h : &&float, uk_d : &&float, numpixels : int)
  C.memcpy(@output_h, @uk_d, numpixels * sizeof(float))
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
  local kernels = {kernelGrad = rawlist[1], kernelUkp1 = rawlist[2]}
  return kernels
end
backend.makeKernelList = makeKernelList



return backend


