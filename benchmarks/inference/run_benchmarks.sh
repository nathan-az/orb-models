#!/bin/bash

set -euxo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

steps=50
stress=false
enable_mlflow=true
mlflow_experiment="orbmol-v2-inference-comparison"
base_args="steps=$steps stress=$stress enable_mlflow=$enable_mlflow mlflow_experiment=$mlflow_experiment precision=high"

torch_compile=(true false)

python $here/bench_md_jax.py n_side=8 $base_args run_name="jax-base" precision=high
python $here/bench_md_jax.py n_side=8 $base_args run_name="jax-base-checkpoint=full" checkpoint=full precision=high
python $here/bench_md_jax.py n_side=11 $base_args run_name="jax-large-checkpointed" checkpoint=full precision=high

for torch_compile_flag in "${torch_compile[@]}"; do
    python $here/bench_md_torch.py $base_args compile=$torch_compile_flag n_side=8 run_name="torch-base-compile-$torch_compile_flag"
done

# Activation checkpointing on the torch backbone (non-reentrant), mirroring the JAX arm.
python $here/bench_md_torch.py $base_args n_side=8 checkpoint=non-reentrant run_name="torch-base-checkpoint=non-reentrant"
python $here/bench_md_torch.py $base_args n_side=8 checkpoint=non-reentrant compile=true run_name="torch-base-checkpoint=non-reentrant-compile"