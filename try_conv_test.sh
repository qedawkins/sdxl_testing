#iree-compile base_ir/conv.mlir \
iree-compile base_ir/conv_1x1.mlir \
    --iree-llvmcpu-target-triple=x86_64-unknown-linux \
    --iree-rocm-target-chip=gfx942 \
    --iree-hal-target-backends=rocm \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-rocm-link-bc=true \
    --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
    -o tmp/conv.vmfb

iree-run-module \
    --module=tmp/conv.vmfb \
    --device=rocm \
    --function=forward \
    --input=2x34x34x1280xf16=1.0 \
    --input=3x3x1280x1280xf16=1.0 \
    --output=@conv_output.npy
    #--input=2x32x32x1280xf16=1.0 \
    #--input=1x1x1280x1280xf16=1.0 \

#iree-compile base_ir/conv.mlir \
iree-compile base_ir/conv_1x1.mlir \
    --iree-llvmcpu-target-triple=x86_64-unknown-linux \
    --iree-rocm-target-chip=gfx942 \
    --iree-hal-target-backends=rocm \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-codegen-transform-dialect-library=specs/attention_and_matmul_spec.mlir \
    --iree-rocm-link-bc=true \
    --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
    -o tmp/conv.vmfb

iree-run-module \
    --module=tmp/conv.vmfb \
    --device=rocm \
    --function=forward \
    --input=2x34x34x1280xf16=1.0 \
    --input=3x3x1280x1280xf16=1.0 \
    --expected_output=2x32x32x1280xf32=@conv_output.npy
    #--input=2x32x32x1280xf16=1.0 \
    #--input=1x1x1280x1280xf16=1.0 \
