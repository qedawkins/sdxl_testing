!image_type = tensor<2x34x34x1280xf16>
!filter_type = tensor<3x3x1280x1280xf16>
!acc_type = tensor<2x32x32x1280xf32>
!result_type = tensor<2x32x32x1280xf16>

#map = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  func.func @forward(%arg0: !image_type, %arg1: !filter_type) -> !result_type {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : !acc_type
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%0 : !acc_type) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> !acc_type
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : !image_type, !filter_type) outs(%1 : !acc_type) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %5 = arith.extf %in : f16 to f32
      %6 = arith.extf %in_0 : f16 to f32
      %7 = arith.mulf %5, %6 : f32
      %8 = arith.addf %out, %7 : f32
      linalg.yield %8 : f32
    } -> !acc_type
    %3 = tensor.empty() : !result_type
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : !acc_type) outs(%3 : !result_type) {
    ^bb0(%in: f32, %out: f16):
      %5 = arith.truncf %in : f32 to f16
      linalg.yield %5 : f16
    } -> !result_type
    return %4 : !result_type
  }
}

