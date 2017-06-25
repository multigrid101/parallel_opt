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
  var numpixels : int = width*height

  var lam : float = 0.0f -- penalty for graddient of image
  var tau : float = 0.1f -- stepsize

  -- START load_input (bleibt immer gleich???)
  -- allocate space for input picture on host and device
  var input_h : &float
  input_h = [&float](C.malloc(numpixels * sizeof(float)))

  var output_h : &float 
  output_h = [&float](C.malloc(numpixels * sizeof(float)))

  -- initialize input picture and u0 and copy to device.
  -- at least the  initialization does not depend on the backend
  for y = 0, height do
    for x = 0, width do
      var center : int = y*width + x
      input_h[center] = 0.5f
      output_h[center] = ([float](C.rand()) / [float](C.RAND_MAX))
    end
  end
  -- END load_input

  -- START allocation
  var input_d : &float
  var gradE_d : &float
  var ukp1_d : &float
  var uk_d : &float

  backend.allocation(&input_d, &gradE_d, &ukp1_d, &uk_d, numpixels)
  -- END allocation


  -- START initialization
  backend.initialization(&input_d, &input_h, &uk_d, &output_h, numpixels)
  -- END initialization


  -- START launch preparation
  var theLaunchParams = backend.launchPreparation(width, height)
  -- END launch preparation
 


  for k = 0,10 do
    C.printf("before: %f\n", output_h[k+width])
  end

  -- main loop
  -- is the same for all backends (except for the kernel launch and possibly some synchronization)
  var maxiter = 200
  for iter = 1,maxiter do

    -- START launch
    backend.launchKernelGrad(theLaunchParams, &gradE_d, lam, &uk_d, &input_d, kernels.kernelGrad)
    backend.launchKernelUkp1(theLaunchParams, &ukp1_d, tau, &uk_d, &gradE_d, kernels.kernelUkp1)
    -- END launch

    -- C.printf("\n")
    -- for k = 0,10 do
    --   C.printf("grad: %f\n", gradE[k+width])
    -- end

    var tmp : &float = uk_d
    uk_d = ukp1_d
    ukp1_d = tmp
  end

  -- copy result back to host
  -- let's put this in a post-optimization block, which would be empty for e.g. the serial backend
  -- START retrieval
  backend.retrieval(&output_h, &uk_d, numpixels)
  -- END retrieval

  -- write result to file TODO
  -- this block will not depends on the backend, but let's abstract it just in cse
  -- future backends have e.g. a different file I/O API
  for k = 0,10 do
    C.printf("after: %f\n", output_h[k+width])
  end
  
end
main()
