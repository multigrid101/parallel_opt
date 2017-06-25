conf = require('grad_descent_config')
height = conf.height
width = conf.width
backend = require(conf.backend)

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
      C.vprintf('%d\n', [&int8](x))

    grad[center] = (uk[center] - input[center]) - lam*((-4)*uk[center] + uk[top] + uk[bottom] + uk[left] + uk[right])
end

terra evalGrad(problemData : &backend.VECDATA)
  var grad = (@problemData).gradE_d
  var lam = (@problemData).lam
  var uk = (@problemData).uk_d
  var input = (@problemData).input_d
  C.vprintf('asdfasdfasdfasdf\n', nil)
  [backend.makeloop(evalGradLocal, {grad, lam, uk, input})]
end
print(evalGrad)


terra evalUkp1Local(idx : backend.Index, ukp1 : &float, tau : float, uk : &float, grad : &float)
      var x = idx.x
      var y = idx.y

      C.vprintf('%d\n', [&int8](x))
      var center : int = y*width + x
      ukp1[center] = uk[center] - tau*grad[center]
      ukp1[center] = 0.0f
      uk[center] = 0.0f
end


terra evalUkp1(problemData : &backend.VECDATA)
  var ukp1 = (@problemData).ukp1_d
  var tau = (@problemData).tau
  var uk = (@problemData).uk_d
  var grad = (@problemData).gradE_d
  C.vprintf('asdfasdfasdfasdf\n', nil)
  [backend.makeloop(evalUkp1Local, {ukp1, tau, uk, grad})]
end
print(evalUkp1)

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
  var maxiter = 100
  for iter = 1,maxiter do

    -- START launch
    backend.launchKernelGrad(theLaunchParams, &problemData, kernels.kernelGrad)
    -- C.sync()
    C.cudaDeviceSynchronize()
    backend.launchKernelUkp1(theLaunchParams, &problemData, kernels.kernelUkp1)
    -- C.sync()
    C.cudaDeviceSynchronize()
    -- END launch

    -- C.printf("\n")
    -- for k = 0,10 do
    --   C.printf("grad: %f\n", problemData.gradE_d[k+width]) -- does not work on GPU
    --   C.printf("iter: %d\n", iter)
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
