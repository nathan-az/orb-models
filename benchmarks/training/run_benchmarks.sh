#!/bin/bash

# Benchmarks with node/edge/graph maxes for packing set to a budget that fits the
# target GPU (these defaults were sized for an RTX 3080, 10 GB).
#   jax is jit'd so pays for packing/padding, including some wasted compute
#   torch compile does not work for training, but then has no wasted compute
#
# Precision-matched throughput comparison: torch vs jax(jvp) vs jax(reverse), all
# in fp32 with NO fp64 in the network -- isolating the framework gap from the
# precision differences that muddied earlier runs. torch=float32-high (TF32) and
# jax matmul_precision=high are the same TF32 math (orb's default), so compute aligns.
# Short run (10 steps); warmup_steps drops JIT-compile / allocator-growth steps
# from the median, so compare median_step_time + peak_mem_gb across the three.
#
# Usage: pass the dataset path as the first arg or via DATA_PATH:
#   ./run_basic_benchmarks.sh /path/to/ase_sqlite.db
#   DATA_PATH=/path/to/ase_sqlite.db ./run_basic_benchmarks.sh

set -euxo pipefail

data_path="${1:-${DATA_PATH:?set DATA_PATH or pass the dataset path as the first argument}}"

# Resolve script dir so this runs from anywhere.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

max_steps=50
warmup_steps=3
edge_budget=16000
node_budget=260
graph_budget=16
mlflow_experiment="orb-v3-train-comparison"

shared_args="max_steps=$max_steps warmup_steps=$warmup_steps data_path=$data_path mlflow_experiment=$mlflow_experiment"
base_budget_args="edge_budget=$edge_budget node_budget=$node_budget graph_budget=$graph_budget"

python "$here/train_torch.py" precision=float32-high $shared_args $base_budget_args run_name="torch-base"
python "$here/train_jax.py" method=jax_reverse matmul_precision=high $shared_args $base_budget_args run_name="jax-reverse-base"
python "$here/train_jax.py" method=jax_jvp matmul_precision=high $shared_args $base_budget_args run_name="jax-jvp-base"
# to show slowdown from chunking, although it allows much larger inputs
python "$here/train_jax.py" method=jax_jvp matmul_precision=high $shared_args $base_budget_args layer_chunk_size=8192 chunk_encoder=true run_name="jax-jvp-base-chunk-8192"

# 4x above, torch cannot handle, reverse also fails
large_budget_args="edge_budget=64000 node_budget=1040 graph_budget=190"

python "$here/train_jax.py" method=jax_jvp matmul_precision=high $shared_args $large_budget_args layer_chunk_size=8192 chunk_encoder=true run_name="jax-jvp-large-chunk-8192"
python "$here/train_jax.py" method=jax_reverse matmul_precision=high $shared_args $large_budget_args run_name="jax-reverse-large-chunk-8192"
