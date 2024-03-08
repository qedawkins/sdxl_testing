for NUM in {0..2500}
do
  FNAME_BASE="$(find sources -name *_main_dispatch_${NUM}.mlir)"
  if (($(echo $FNAME_BASE | grep -c . ) == 1)); then
    echo "Testing $FNAME_BASE"
    ~/iree-build/tools/iree-compile \
        --iree-input-type=none \
        --iree-rocm-target-chip=gfx942 \
        --iree-rocm-link-bc=true \
        --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
        --iree-hal-target-backends=rocm \
        --iree-rocm-link-bc=true \
        --iree-opt-const-eval=false \
        --iree-codegen-llvmgpu-use-vector-distribution \
        --iree-codegen-transform-dialect-library=specs/attention_mfma_transform_64_spec.mlir \
        --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
        --compile-from=executable-sources \
        --mlir-disable-threading \
        $FNAME_BASE \
        -o /tmp/module.vmfb

  else
    echo "Could not find single source for ${NUM}"
  fi
done
