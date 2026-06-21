# JAX vs torch benchmarks

Throughput / memory benchmarks for the **JAX port** of the orb forcefields against the **torch** reference, for both **training** and **inference**. The JAX model code lives in the library (`orb_models/{forcefield,common}/models/jax`); this directory is only the harness. Numbers below are from the bundled `run_benchmarks.sh` scripts and live in MLflow (experiments `orb-v3-train-comparison` and `orbmol-v2-inference-comparison`).

## Motivation

The port exists to get two things torch does not give us on a single 10 GB GPU:

1. **Whole-graph XLA fusion.** `jit` compiles the entire step (geometry → message passing → energy → force autodiff, plus PME at inference) into one optimised program. This is faster and leaner per step than torch's eager / `torch.compile` execution.
2. **Composable autodiff + rematerialisation as memory levers.** JAX lets us change *how* the force/Hessian derivatives are taken and *what* gets recomputed in the backward pass, trading a slower step for a much smaller peak — which lets JAX fit inputs that the torch reference simply OOMs on.

Each benchmark therefore has two parts: a **raw framework comparison** at a base input size both frameworks can run, then a **memory lever** shown twice — at the base size (where it costs throughput) and at a larger size (which torch cannot fit at all).

The key lever differs by workload:

- **Training** — the loss needs a *second* derivative (energy → force is the first; the loss-grad is the second). Forward-over-reverse (**`jax_jvp`**) plus **edge chunking** tiles the dominant per-edge second-derivative buffers. Reverse-mode is faster at the base size but cannot be chunked (XLA already auto-remats it), so it OOMs at scale.
- **Inference** — forces are a single VJP, so the lever is plain **activation checkpointing** of the GNN stacks, recomputing them in the force backward.

## Setup

- **Hardware:** RTX 3080, 10 GB. **Precision:** fp32 with TF32 matmuls (`high`) on both frameworks — orb's loader default; pass `precision=highest` (and `matmul_precision=highest`) for true fp32.
- torch and JAX cannot share a CUDA context in one process, so every run is **one framework per process**. Weights are random at released dims (a shape / memory / throughput question, not a parity one).
- Run: `training/run_benchmarks.sh /path/to/ase.db` and `inference/run_benchmarks.sh`.

A note on metric names (they differ between the two harnesses): training logs **`median_step_time`** (seconds) and **`peak_mem_gb`**; inference logs **`force_median_ms`** (device step, ms), **`prep_median_ms`** (host neighbour-list build, ms) and **`peak_mem_mib`**. `atoms/s` is derived here (real atoms in the measured step ÷ step time) and is not stored in MLflow.

---

## Training

One finetuning step = forward + double-backward (forces), bucketed FFD-packed data. JAX `jit`s to a **fixed padded budget** every step (else XLA recompiles), so it pays the padded-shape compute regardless of real content; torch runs the packed bucket unpadded. Compared at a matched base budget (`16000` edges / `260` nodes / `16` graphs), then at a 4× budget.

**Base budget** (measured step ≈ 152 atoms / 9k edges):

| Run | grad path | lever | step (ms) | peak (GB) | atoms/s |
|---|---|---|---|---|---|
| `torch-base` | reverse (double-backward) | — | 242 | 7.9 | 630 |
| `jax-reverse-base` | reverse | — | **146** | 4.5 | **1040** |
| `jax-jvp-base` | jvp | — | 192 | 4.6 | 790 |
| `jax-jvp-base-chunk-8192` | jvp | edge-chunk 8192 | 260 | **1.4** | 580 |

Both JAX paths beat torch (~1.3–1.7× faster, ~1.7× leaner) despite paying for padding. Reverse is the fastest base step; the jvp **edge-chunk** lever cuts peak memory 4.6 → 1.4 GB (3.2×) at a ~1.4× step-time cost.

**Large budget** (`64000` / `1040` / `190`, ≈ 952 atoms / 62k edges — 4× the base):

| Run | grad path | lever | step (ms) | peak (GB) | atoms/s |
|---|---|---|---|---|---|
| `jax-jvp-large-chunk-8192` | jvp | edge-chunk 8192 | 993 | 2.9 | 960 | 
| `jax-reverse-large` | reverse | — | — | — | **OOM** |

This is the payoff: jvp + chunking trains a bucket **4× larger** than the base, which torch cannot reach (it OOMs at ~16 k edges ≈ the base budget) and which reverse-mode JAX also OOMs on (chunking it just adds overhead on top of XLA's auto-remat). This also recovers relative speed (atoms/sec) while unlocking training on larger examples.

---

## Inference

The MD inner loop on synthetic aqueous NaCl(aq) with **OrbMol-v2** (periodic PME). Atom count is fixed and only the neighbour list shifts as atoms move, so JAX `jit`-compiles once and reuses it. The lever is **`checkpoint=full`** (remat the GNN stacks in the force backward).

**Base size** (`n_side=8`, 1476 atoms):

| Run | lever | force step (ms) | peak (MiB) | atoms/s |
|---|---|---|---|---|
| `torch-base-compile-false` | — | 135 | 4209 | 10900 |
| `torch-base-compile-true` | — | 156 | 4046 | 9450 |
| `jax-base` | — | **125** | 2609 | **11800** |
| `jax-base-checkpoint=full` | ckpt=full | 187 | **992** | 7900 |

JAX's device step is the fastest and ~1.6× leaner than torch; `torch.compile` does not help at this size (it is slower). The checkpoint lever trades ~1.5× step time for **2.6× less memory** (2609 → 992 MiB). (Host neighbour-list `prep` is higher for JAX — ~33 vs ~14 ms — so wall-clock totals are closer; `prep` is a CPU harness artifact, the device `force` step is the framework comparison.)

**Large size** (`n_side=11`, 3837 atoms):

| Run | lever | force step (ms) | peak (MiB) | atoms/s |
|---|---|---|---|---|
| `jax-large-checkpointed` | ckpt=full | 476 | 2406 | 8060 |

3837 atoms fits in **2.4 GB** with checkpointing — a system size neither torch nor un-checkpointed JAX fits on the 10 GB card (the un-checkpointed ceiling is ~2880 atoms). The lever turns memory headroom into reachable system size at a predictable per-step cost.
