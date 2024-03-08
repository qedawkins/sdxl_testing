module @sdxl_compiled_pipeline {
  func.func private @compiled_unet.main(%arg0: tensor<1x4x128x128xf16>, %arg1: tensor<2x64x2048xf16>, %arg2: tensor<2x1280xf16>, %arg3: tensor<2x6xf16>, %arg4: tensor<1xf16>, %arg5: tensor<1xi64>) -> tensor<1x4x128x128xf16> attributes { iree.abi.model = "coarse-fences" }
  
  func.func @forward(%arg0: i64) -> tensor<1x4x128x128xf16> {
    %sample = arith.constant dense<0.000000e+00> : tensor<1x4x128x128xf16>
    %p_embeds = arith.constant dense<0.000000e+00> : tensor<2x64x2048xf16>
    %t_embeds = arith.constant dense<0.000000e+00> : tensor<2x1280xf16>
    %time_ids = arith.constant dense<0.000000e+00> : tensor<2x6xf16>
    %guidance_scale = arith.constant dense<0.000000e+00> : tensor<1xf16>

    util.optimization_barrier %sample : tensor<1x4x128x128xf16>
    util.optimization_barrier %p_embeds : tensor<2x64x2048xf16>
    util.optimization_barrier %t_embeds : tensor<2x1280xf16>
    util.optimization_barrier %time_ids : tensor<2x6xf16>
    util.optimization_barrier %guidance_scale : tensor<1xf16>

    %step_64 = arith.index_cast %arg0 : i64 to index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %res = scf.for %arg1 = %c0 to %step_64 step %c1 iter_args(%arg = %sample) -> (tensor<1x4x128x128xf16>) {
      %step = arith.index_cast %arg1 : index to i64
      %this_step = tensor.from_elements %step : tensor<1xi64>
      %inner = func.call @compiled_unet.main(%arg, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %this_step) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
      scf.yield %inner : tensor<1x4x128x128xf16>
    }
    return %res : tensor<1x4x128x128xf16>
  } 
}
