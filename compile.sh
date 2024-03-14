#iree-compile base_ir/scheduled_unet.mlir \
iree-compile base_ir/generated_unet.mlir \
    --iree-vulkan-target-triple=rdna3-unknown-linux \
    --iree-llvmcpu-target-triple=x86_64-unknown-linux \
    --iree-hal-cuda-llvm-target-arch=sm_80 \
    --iree-rocm-target-chip=gfx942 \
    --iree-hal-target-backends=rocm \
    --iree-global-opt-propagate-transposes=true \
    --iree-opt-outer-dim-concat=true \
    --iree-rocm-link-bc=true \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
    --iree-execution-model=async-external \
    --iree-hal-dump-executable-configurations-to=configurations \
    --iree-hal-dump-executable-sources-to=sources \
    --iree-hal-dump-executable-binaries-to=binaries \
    --iree-hal-dump-executable-benchmarks-to=benchmarks \
    --iree-opt-splat-parameter-archive-export-file=tmp/unet.irpa \
    --iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-transpose-convolution-pipeline)" \
    --iree-codegen-transform-dialect-library=specs/attention_and_matmul_spec.mlir \
    --mlir-disable-threading \
    -o tmp/unet.mlir
    #--mlir-print-ir-after=iree-stream-schedule-concurrency \
    #--iree-global-opt-only-sink-transposes=true \
    #--compile-to=global-optimization \
    #-o tmp/unet.mlir
    #-o tmp/scheduled_unet.vmfb

iree-benchmark-module \
    --module=tmp/unet.vmfb \
    --device=rocm://5 \
    --function=main \
    --parameters=model=tmp/unet.irpa \
    --input=1x4x128x128xf16 \
    --input=1xi64 \
    --input=2x64x2048xf16 \
    --input=2x1280xf16 \
    --input=2x6xf16 \
    --input=1xf16 \
    --device_allocator=caching
