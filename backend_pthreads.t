backend = {}

C = terralib.includecstring([[
#include <pthread.h>
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
  numpixels : int,
  tid : int
}
backend.VECDATA = VECDATA


struct Index { 
               x : int,
               y : int 
             }
terra Index:initAndTestIfValid(width : int, height : int, x: int, y: int)
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


terra launchKernelGrad(launch : bool, problemData : &backend.VECDATA, kernel : {&opaque} -> &opaque)
  var data_t1 : VECDATA
  var data_t2 : VECDATA

  data_t1.gradE_d = (@problemData).gradE_d
  data_t1.lam = (@problemData).lam
  data_t1.uk_d = (@problemData).uk_d
  data_t1.input_d = (@problemData).input_d
  data_t1.tid = 0
  data_t1.numpixels = width*height

  data_t2.gradE_d = (@problemData).gradE_d
  data_t2.lam = (@problemData).lam
  data_t2.uk_d = (@problemData).uk_d
  data_t2.input_d = (@problemData).input_d
  data_t2.tid = 1
  data_t2.numpixels = width*height

  var t1 : C.pthread_t
  var t2 : C.pthread_t

  -- var data_t1 = [&VECDATA](launch)[0]
  -- var data_t2 = [&VECDATA](launch)[1]

  C.pthread_create(&t1, nil, kernel, &data_t1)
  C.pthread_create(&t2, nil, kernel, &data_t2)
end
backend.launchKernelGrad = launchKernelGrad


terra launchKernelUkp1(launch : bool, problemData : &backend.VECDATA, kernel : {&opaque} -> &opaque)
  var data_t1 : VECDATA
  var data_t2 : VECDATA

  data_t1.ukp1_d = (@problemData).ukp1_d
  data_t1.tau = (@problemData).tau
  data_t1.uk_d = (@problemData).uk_d
  data_t1.gradE_d = (@problemData).gradE_d
  data_t1.tid = 0
  data_t1.numpixels = width*height

  data_t2.ukp1_d = (@problemData).ukp1_d
  data_t2.tau = (@problemData).tau
  data_t2.uk_d = (@problemData).uk_d
  data_t2.gradE_d = (@problemData).gradE_d
  data_t2.tid = 1
  data_t2.numpixels = width*height

  var t1 : C.pthread_t
  var t2 : C.pthread_t

  -- var data_t1 = [&VECDATA](launch)[0]
  -- var data_t2 = [&VECDATA](launch)[1]

  C.pthread_create(&t1, nil, kernel, &data_t1)
  C.pthread_create(&t2, nil, kernel, &data_t2)

  C.pthread_join(t1, nil)
  C.pthread_join(t2, nil)
end
backend.launchKernelUkp1 = launchKernelUkp1


terra retrieval(problemData : &backend.VECDATA)
  var numpixels = (@problemData).numpixels
  C.memcpy((@problemData).output_h, (@problemData).uk_d, numpixels * sizeof(float))
end
backend.retrieval = retrieval


backend.tid_sym = symbol(int, "tid")
function makeloop(func, args)
  return quote
    for y = 0,width do
      for x = [backend.tid_sym]*(height/2), [backend.tid_sym]*(height/2) + (height/2) do
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
  local kernels = {}
  local rawGrad = rawlist[1]
  local rawUkp1 = rawlist[2]
  kernels.kernelGrad = terra(arg : &opaque) : &opaque
    var thedata = @([&backend.VECDATA](arg))

    var grad = thedata.gradE_d
    var lam = thedata.lam
    var uk = thedata.uk_d
    var input = thedata.input_d
    var tid = thedata.tid

    rawGrad(grad, lam, uk, input, tid)
    return nil
  end
  kernels.kernelUkp1 = terra(arg : &opaque) : &opaque
    var thedata = @([&backend.VECDATA](arg))

    var ukp1 = thedata.ukp1_d
    var tau = thedata.tau
    var uk = thedata.uk_d
    var grad = thedata.gradE_d
    var tid = thedata.tid

    evalUkp1(ukp1, tau, uk, grad, tid)
    return nil
  end

  return kernels
end
backend.makeKernelList = makeKernelList

return backend


