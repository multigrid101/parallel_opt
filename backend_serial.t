backend = {}
backend.tid_sym = symbol(int, "dummy_not_required_for_this_backend")

C = terralib.includecstring([[
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
]])


struct VECDATA {
  gradE_d : &float,
  uk_d : &float,
  ukp1_d : &float,
  input_d : &float,
  input_h : &float,
  output_h : &float,
  tau : float,
  lam : float,
  w : int,
  h : int
  numpixels : int
}
backend.VECDATA = VECDATA


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


terra allocation(problemData : &backend.VECDATA)
  var numpixels = (@problemData).numpixels

  (@problemData).input_d = [&float](C.malloc(numpixels * sizeof(float)))
  (@problemData).gradE_d = [&float](C.malloc(numpixels * sizeof(float)))
  (@problemData).ukp1_d = [&float](C.malloc(numpixels * sizeof(float)))
  (@problemData).uk_d = [&float](C.malloc(numpixels * sizeof(float)))
end
backend.allocation = allocation


terra initialization(problemData : &backend.VECDATA)
  var numpixels = (@problemData).numpixels
  C.memcpy((@problemData).input_d, (@problemData).input_h, numpixels * sizeof(float))
  C.memcpy((@problemData).uk_d, (@problemData).output_h, numpixels * sizeof(float))
end
backend.initialization = initialization

terra launchPreparation(width : int, height : int)
  return true
end
backend.launchPreparation = launchPreparation


terra launchKernelGrad(launch: bool, problemData : &backend.VECDATA, kernel : {&float, float, &float, &float, int} -> {})
  kernel((@problemData).gradE_d, (@problemData).lam, (@problemData).uk_d, (@problemData).input_d, 0) -- '0' is just a dummy parameter here
end
backend.launchKernelGrad = launchKernelGrad


terra launchKernelUkp1(launch : bool, problemData : &backend.VECDATA, kernel : {&float, float, &float, &float, int} -> {})
  kernel((@problemData).ukp1_d, (@problemData).tau, (@problemData).uk_d, (@problemData).gradE_d, 0) -- '0' is just a dummy parameter here
end
backend.launchKernelUkp1 = launchKernelUkp1


terra retrieval(problemData : &backend.VECDATA)
  var numpixels = (@problemData).numpixels
  C.memcpy((@problemData).output_h, (@problemData).uk_d, numpixels * sizeof(float))
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


