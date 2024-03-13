#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @forward(%arg0: tensor<8192x640xf16>, %arg1: tensor<640x640xf16>, %arg2: tensor<640xf32>, %arg3: tensor<8192x640xf16>) -> tensor<8192x640xf16> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<8192x640xf16>
    %1 = tensor.empty() : tensor<8192x640xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<8192x640xf32>) -> tensor<8192x640xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8192x640xf16>, tensor<640x640xf16>) outs(%2 : tensor<8192x640xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %5 = arith.extf %in : f16 to f32
      %6 = arith.extf %in_0 : f16 to f32
      %7 = arith.mulf %5, %6 : f32
      %8 = arith.addf %out, %7 : f32
      linalg.yield %8 : f32
    } -> tensor<8192x640xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map4, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%3, %arg2, %arg3 : tensor<8192x640xf32>, tensor<640xf32>, tensor<8192x640xf16>) outs(%0 : tensor<8192x640xf16>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f16, %out: f16):
      %5 = arith.addf %in, %in_0 : f32
      %6 = arith.truncf %5 : f32 to f16
      %7 = arith.addf %6, %in_1 : f16
      linalg.yield %7 : f16
    } -> tensor<8192x640xf16>
    return %4 : tensor<8192x640xf16>
  }
}

