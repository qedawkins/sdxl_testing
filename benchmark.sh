for NUM in {1..10}
do
  FNAME_BASE="$(find benchmarks -name *async_dispatch_${NUM}_*)"
  if (($(echo $FNAME_BASE | grep -c . ) == 1)); then
    echo "Testing $FNAME_BASE"
    iree-compile \
        --iree-rocm-target-chip=gfx942 \
        --iree-rocm-link-bc=true \
        --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
        --iree-hal-target-backends=rocm \
        --iree-rocm-link-bc=true \
        --iree-opt-const-eval=false \
        --iree-codegen-llvmgpu-use-vector-distribution \
        --iree-codegen-transform-dialect-library=specs/attention_mfma_transform_64_spec.mlir \
        --iree-hal-benchmark-dispatch-repeat-count=100 \
        --compile-from=executable-sources \
        $FNAME_BASE \
        -o tmp/module.vmfb

    FUNCTION="$(cat $FNAME_BASE | grep iree.benchmark | grep -o -o '[^ @]*_dispatch[^ (]*')"
    echo "Compiled, trying with function $FUNCTION"
    
    iree-benchmark-module \
        --module=tmp/module.vmfb \
        --function=$FUNCTION \
        --input=1 \
        --batch_size=100 \
        --device=vulkan
  else
    echo "Could not find single benchmark for ${NUM}"
  fi
done
