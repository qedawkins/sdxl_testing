module @sdxl_compiled_pipeline {
  func.func private @compiled_unet.main(%arg0: tensor<1x4x128x128xf16>, %arg5: tensor<1xi64>, %arg1: tensor<2x64x2048xf16>, %arg2: tensor<2x1280xf16>, %arg3: tensor<2x6xf16>, %arg4: tensor<1xf16>) -> tensor<1x4x128x128xf16> attributes { iree.abi.model = "coarse-fences" }
  
  func.func @forward(%sample: tensor<1x4x128x128xf16>, %this_step: tensor<1xi64>, %p_embeds: tensor<2x64x2048xf16>, %t_embeds: tensor<2x1280xf16>, %time_ids: tensor<2x6xf16>, %guidance_scale: tensor<1xf16>) -> tensor<1x4x128x128xf16> {
    %inner = func.call @compiled_unet.main(%sample, %this_step, %p_embeds, %t_embeds, %time_ids, %guidance_scale) : (tensor<1x4x128x128xf16>, tensor<1xi64>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>) -> tensor<1x4x128x128xf16>
    return %inner : tensor<1x4x128x128xf16>
  } 
}
