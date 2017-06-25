-- backend = require('backend_cuda')
backend = require('backend_serial')

height = 100
width = 200
terra evalGradLocal(idx : backend.Index, grad : &float, lam : float, uk : &float, input : &float)
    var x = idx.x
    var y = idx.y

    var center : int = y*width + x
    var top : int = (y-1)*width + x
    var bottom : int = (y+1)*width + x
    var left : int = y*width + (x-1)
    var right : int = y*width + (x+1)

    -- test if whole stencil is inside image
    var isinside : bool = (x>=1) and (x<= width-2) and (y>=1) and (y<= height-2)

    -- if stencil is not completely inside image, then clamp all indices that are outside of the image
    if not isinside then
      if (y-1) < 0 then top = y*width + x end
      if (y+1) > height-1 then bottom = y*width + x end
      if (x-1) < 0 then left = y*width + x end
      if (x+1) > width-1 then right = y*width + x end
    end

    grad[center] = (uk[center] - input[center]) - lam*((-4)*uk[center] + uk[top] + uk[bottom] + uk[left] + uk[right])
end

terra evalGrad(grad : &float, lam : float, uk : &float, input : &float)
  [backend.makeloop(evalGradLocal, {grad, lam, uk, input})]
end


terra evalUkp1Local(idx : backend.Index, ukp1 : &float, tau : float, uk : &float, grad : &float)
      var x = idx.x
      var y = idx.y

      var center : int = y*width + x
      ukp1[center] = uk[center] - tau*grad[center]
end


terra evalUkp1(ukp1 : &float, tau : float, uk : &float, grad : &float)
  [backend.makeloop(evalUkp1Local, {ukp1, tau, uk, grad})]
end

-- TODO need to do something kernel-specific here. Maybe try to make this the
-- spot where we wrap the serial serial kernel-code into a for-loop
-- local kernels = terralib.cudacompile({kernelGrad = evalGrad, kernelUkp1 = evalUkp1})
local kernels = backend.makeKernelList({evalGrad, evalUkp1})


terra main()
  var problemData : backend.VECDATA

  var numpixels : int = width*height
  problemData.numpixels = numpixels
  problemData.w = width
  problemData.h = height


  problemData.lam = 1.0f
  problemData.tau = 0.1f

  -- START load_input (bleibt immer gleich???)
  -- allocate space for input picture on host and device
  problemData.input_h = [&float](C.malloc(numpixels * sizeof(float)))
  problemData.output_h = [&float](C.malloc(numpixels * sizeof(float)))


  -- initialize input picture and u0 and copy to device.
  -- at least the  initialization does not depend on the backend
  for y = 0, height do
    for x = 0, width do
      var center : int = y*width + x
      problemData.input_h[center] = 0.5f
      problemData.output_h[center] = ([float](C.rand()) / [float](C.RAND_MAX))
    end
  end
  -- END load_input

  -- START allocation
  backend.allocation(&problemData)
  -- END allocation


  -- START initialization
  backend.initialization(&problemData)
  -- END initialization


  -- START launch preparation
  var theLaunchParams = backend.launchPreparation(width, height)
  -- END launch preparation
 


  for k = 0,10 do
    C.printf("before: %f\n", problemData.output_h[k+width])
  end

  -- main loop
  -- is the same for all backends (except for the kernel launch and possibly some synchronization)
  var maxiter = 200
  for iter = 1,maxiter do

    -- START launch
    backend.launchKernelGrad(theLaunchParams, &problemData, kernels.kernelGrad)
    backend.launchKernelUkp1(theLaunchParams, &problemData, kernels.kernelUkp1)
    -- END launch

    -- C.printf("\n")
    -- for k = 0,10 do
    --   C.printf("grad: %f\n", problemData.gradE_d[k+width])
    -- end

    var tmp : &float = problemData.uk_d
    problemData.uk_d = problemData.ukp1_d
    problemData.ukp1_d = tmp
  end

  -- copy result back to host
  -- let's put this in a post-optimization block, which would be empty for e.g. the serial backend
  -- START retrieval
  backend.retrieval(&problemData)
  -- END retrieval

  -- write result to file TODO
  -- this block will not depends on the backend, but let's abstract it just in cse
  -- future backends have e.g. a different file I/O API
  for k = 0,10 do
    C.printf("after: %f\n", problemData.output_h[k+width])
    -- C.printf("after: %f\n", problemData.uk_d[k+width])
  end
  
end
main()
