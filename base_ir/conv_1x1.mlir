module {
  func.func @forward(%lhs: tensor<2x32x32x1280xf16>, %rhs: tensor<1x1x1280x1280xf16>) -> tensor<2x32x32x1280xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %1 = tensor.empty() : tensor<2x32x32x1280xf32>
    %out = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
    %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
      ins(%lhs, %rhs : tensor<2x32x32x1280xf16>, tensor<1x1x1280x1280xf16>)
      outs(%out : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>

    return %13 : tensor<2x32x32x1280xf32>
  }
}

