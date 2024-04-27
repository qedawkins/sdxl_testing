// Transform dialect specification for attention on MI300 with MFMA.
// This script only supports variants of attention with a sequence
// length that is a multiple of 64. There are two near duplicate
// because we need different tile sizes when the head dimension is 512.
// TODO: Figure out how to parameterize the tile sizes without duplicating
// the attention function.

#layout_16 = #iree_gpu.mfma_layout<F16_16x16x16_F32>
#layout = #iree_gpu.mfma_layout<F16_32x32x8_F32>

!image_type = tensor<2x34x34x1280xf16>
!filter_type = tensor<3x3x1280x1280xf16>
!acc_type = tensor<2x32x32x1280xf32>

!tiled_image_type = tensor<2x20x34x34x64xf16>
!tiled_filter_type = tensor<8x20x3x3x160x64xf16>
!tiled_acc_type = tensor<2x8x32x32x160xf32>

#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#img_map = affine_map<(b, ooc, oic, ih, iw, ioc, kh, kw, iic) -> (b, oic, ih + kh, iw + kw, iic)>
#filter_map = affine_map<(b, ooc, oic, ih, iw, ioc, kh, kw, iic) -> (ooc, oic, kh, kw, ioc, iic)>
#out_map = affine_map<(b, ooc, oic, ih, iw, ioc, kh, kw, iic) -> (b, ooc, ih, iw, ioc)>

module attributes { transform.with_named_sequence } {
//===----------------------------------------------------------------------===//
// Matmul tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}, %config: !transform.any_param {transform.readonly}) {
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    // transform.print %op {name = "Applied"} : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%arg0: !image_type, %arg1: !filter_type, %1: !acc_type):
      %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : !image_type, !filter_type) outs(%1 : !acc_type) {
      ^bb0(%in: f16, %in_0: f16, %out: f32):
        %5 = arith.extf %in : f16 to f32
        %6 = arith.extf %in_0 : f16 to f32
        %7 = arith.mulf %5, %6 : f32
        %8 = arith.addf %out, %7 : f32
        linalg.yield %8 : f32
      } -> !acc_type
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 2, 32, 160, 0, 0, 64], [0, 0, 0, 0, 1, 1, 0]]>,
    translation_info = #iree_codegen.translation_info<LLVMGPUConvVectorDistribute,
      {mma_schedule = #iree_gpu.mma_schedule<
        intrinsic = #iree_gpu.mfma_layout<F16_32x32x8_F32>,
        subgroup_m_count = 2, subgroup_n_count = 5,
        subgroup_m_tile_count = 1,
        subgroup_n_tile_count = 1,
        subgroup_k_tile_count = 8>}>,
      workgroup_size = [640, 1, 1], subgroup_size = 64> -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_tiled_conv2d(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%arg0: !tiled_image_type, %arg1: !tiled_filter_type, %1: !tiled_acc_type):
      %2 = linalg.generic {indexing_maps = [#img_map, #filter_map, #out_map], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : !tiled_image_type, !tiled_filter_type) outs(%1 : !tiled_acc_type) {
      ^bb0(%in: f16, %in_0: f16, %out: f32):
        %5 = arith.extf %in : f16 to f32
        %6 = arith.extf %in_0 : f16 to f32
        %7 = arith.mulf %5, %6 : f32
        %8 = arith.addf %out, %7 : f32
        linalg.yield %8 : f32
      } -> !tiled_acc_type
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
    // lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 2, 32, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0]]>,
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 2, 32, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0]]>,
    translation_info = #iree_codegen.translation_info<LLVMGPUConvVectorDistribute,
      {mma_schedule = #iree_gpu.mma_schedule<
        intrinsic = #iree_gpu.mfma_layout<F16_32x32x8_F32>,
        subgroup_m_count = 2, subgroup_n_count = 5,
        subgroup_m_tile_count = 1,
        subgroup_n_tile_count = 1,
        subgroup_k_tile_count = 8>}>,
      workgroup_size = [640, 1, 1], subgroup_size = 64> -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280 -> @apply_op_config,
        @match_tiled_conv2d -> @apply_op_config
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module
