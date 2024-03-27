iree-compile base_ir/conv.mlir \
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
    --input=@image.npy \
    --input=@filter.npy \
    --output=@conv_output.npy

iree-compile base_ir/conv.mlir \
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
    --input=@image.npy \
    --input=@filter.npy \
    --expected_output=@conv_output.npy
