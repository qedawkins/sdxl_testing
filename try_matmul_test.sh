iree-compile base_ir/matmul_fused.mlir \
    --iree-llvmcpu-target-triple=x86_64-unknown-linux \
    --iree-rocm-target-chip=gfx942 \
    --iree-hal-target-backends=rocm \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-rocm-link-bc=true \
    --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
    -o /tmp/matmul_fused.vmfb

iree-run-module \
    --module=/tmp/matmul_fused.vmfb \
    --device=rocm \
    --function=forward \
    --input=8192x640xf16=1.0 \
    --input=640x640xf16=2.0 \
    --input=640xf32=3.0 \
    --input=8192x640xf16=4.0 \
    --output=@output.npy

iree-run-module \
    --module=/tmp/matmul_fused.vmfb \
    --device=rocm \
    --function=forward \
    --input=8192x640xf16=1.0 \
    --input=640x640xf16=2.0 \
    --input=640xf32=3.0 \
    --input=8192x640xf16=4.0 \
    --expected_output=4xf32=@output.npy
