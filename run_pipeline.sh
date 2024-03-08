iree-compile pipelines/run_unet.mlir \
    --iree-rocm-target-chip=gfx942 \
    --iree-hal-target-backends=rocm \
    --iree-rocm-link-bc=true \
    --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
    --mlir-disable-threading \
    -o tmp/pipeline.vmfb

iree-benchmark-module \
    --module=tmp/unet.vmfb \
    --module=tmp/pipeline.vmfb \
    --device=rocm://5 \
    --function=forward \
    --parameters=model=tmp/unet.irpa \
    --input=1x4x128x128xf16 \
    --input=1xi64 \
    --input=2x64x2048xf16 \
    --input=2x1280xf16 \
    --input=2x6xf16 \
    --input=1xf16 \
    --device_allocator=caching

iree-compile pipelines/loop_run_unet.mlir \
    --iree-rocm-target-chip=gfx942 \
    --iree-hal-target-backends=rocm \
    --iree-rocm-link-bc=true \
    --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
    --mlir-disable-threading \
    -o tmp/pipeline.vmfb

iree-benchmark-module \
    --module=tmp/unet.vmfb \
    --module=tmp/pipeline.vmfb \
    --device=rocm://5 \
    --function=forward \
    --parameters=model=tmp/unet.irpa \
    --input=1x4x128x128xf16 \
    --input=2x64x2048xf16 \
    --input=2x1280xf16 \
    --input=2x6xf16 \
    --input=1xf16 \
    --input=100 \
    --device_allocator=caching
