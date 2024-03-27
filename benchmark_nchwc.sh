iree-compile base_ir/baseline_conv.mlir \
    --iree-rocm-target-chip=gfx942 \
    --iree-hal-target-backends=rocm \
    --iree-rocm-link-bc=true \
    --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
    --iree-execution-model=async-external \
    --iree-codegen-transform-dialect-library=specs/conv_spec.mlir \
    --mlir-disable-threading \
    -o tmp/conv.vmfb

iree-benchmark-module \
    --module=tmp/conv.vmfb \
    --device=rocm://5 \
    --function=forward \
    --input=2x34x34x1280xf16 \
    --input=3x3x1280x1280xf16 \
    --device_allocator=caching

iree-compile base_ir/tiled_conv.mlir \
    --iree-rocm-target-chip=gfx942 \
    --iree-hal-target-backends=rocm \
    --iree-rocm-link-bc=true \
    --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
    --iree-execution-model=async-external \
    --iree-codegen-transform-dialect-library=specs/conv_spec.mlir \
    --mlir-disable-threading \
    --mlir-print-ir-after-all \
    -o tmp/conv.vmfb

iree-benchmark-module \
    --module=tmp/conv.vmfb \
    --device=rocm://5 \
    --function=forward \
    --input=2x20x34x34x64xf16 \
    --input=8x20x3x3x160x64xf16 \
    --device_allocator=caching
