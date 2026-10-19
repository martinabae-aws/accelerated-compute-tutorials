---
title: MoE / Hybrid-EP Validation
description: The layer above EFA — design principles for placing Expert Parallelism (EP) inside a node for large-scale MoE training, and the procedure for validating the Hybrid-EP dispatcher
---

# MoE / Hybrid-EP Validation

This page covers the **model and library layer** that sits on top of EFA networking. After confirming that the network (EFA and NCCL) is healthy, this is the stage where you decide and validate **where to place Expert Parallelism (EP) and which dispatcher backend to use** when training a large-scale MoE (Mixture-of-Experts) model.

There is one central question — **"Will Expert communication (all-to-all) go out over EFA (between nodes), or stay confined to NVLink within a node?"** This choice determines MoE training performance, which is why EP design is the most important decision.

!!! note "Validated environment (as of 2026-10)"
    The versions, commits, commands, and reference values below are a **snapshot from a specific point in time measured on a P6-B200 2-node (16 GPU) environment**. They are reference figures rather than guaranteed-performance values, and will vary with the number of nodes, the model, the parallelism configuration, and software versions. The principles (the main text) hold regardless of version, so when versions change you only need to refresh this box.

    - Container image: `nvcr.io/nvidia/pytorch:26.08-py3` (+ EFA installer 1.50.0)
    - Megatron-LM: 0.19.0
    - Hybrid-EP library: specific commit `94a9f8f6b146c07d97ec58f67cd6d303296d6098` (intra-node-only build)
    - Validation parallelism: EP=8, TP=1, PP=1 · 2 nodes, 16 GPUs

## Why keep EP inside a node

MoE performs **all-to-all** communication to send tokens to each expert and get the results back. Because every rank exchanges data with every other rank, it is far more sensitive to network bandwidth and latency than the all-reduce of data parallelism.

The decisive factor here is the communication path.

- **Inside a node (8 GPUs) it is NVLink** — about 900 GB/s per GPU in one direction.
- **Between nodes it is EFA** — 3.2 Tbps per instance (about 400 GB/s), which divided across 8 GPUs is **about 50 GB/s per GPU**.

In other words, intra-node NVLink and inter-node EFA differ in bandwidth by more than an order of magnitude. **When the EP degree exceeds the number of GPUs in a node (8)**, all-to-all traffic leaves NVLink and goes out over EFA, and from that point on EP communication performance is capped by EFA bandwidth (about 50 GB/s per GPU).

The design principle is therefore clear.

!!! tip "Design principles"
    1. **Keep EP inside a node (≤ 8).** When the EP group fits within a single node, expert communication happens only over NVLink and does not use EFA.
    2. Scale out between nodes with **data parallelism (DP) and pipeline parallelism (PP)**. Between nodes, only the DP communication that reconciles gradients across nodes holding the same experts goes over EFA.
    3. For the EP group to fit within one node, you need **EP × TP ≤ 8**, and the 8 ranks per node must be placed **contiguously** (Megatron groups EP groups from contiguous ranks).

## Communication path comparison (measured reference)

Inter-node all-to-all is already running close to the upper limit of EFA bandwidth, so **the room for improvement is not in tuning but in design (keeping EP inside a node).**

!!! note "Validated environment measurements (P6-B200 2 nodes, 16 GPUs)"
    | Communication | Path | Measured value | Meaning |
    |---|---|---|---|
    | all-reduce (16 GiB, nccl-tests) | Intra-node NVLink + inter-node EFA (NCCL NVLS tree) | algbw **374 GB/s** | About 94% of the 400 GB/s EFA per node (calculated value) |
    | all-to-all (16 GiB, nccl-tests) | Inter-node EFA | algbw **95 GB/s** (about 48 GB/s inter-node traffic per GPU) | About 95% of the roughly 50 GB/s EFA ceiling per GPU (calculated value) |
    | Hybrid-EP dispatch / combine (1 node, 8 GPUs, hidden 7168, 4096 tokens/rank, top-8) | Intra-node NVLink | By API measurement: dispatch 419 GB/s (FP8), combine 512 GB/s | Several to roughly 10× the inter-node all-to-all traffic per GPU (the measurement method differs, so a direct comparison is for reference only) |

    - all-reduce (data parallelism) uses nearly all of the EFA bandwidth.
    - **Inter-node all-to-all also runs up to about 95% of the EFA ceiling** — meaning there is almost no room to push it higher.
    - If the same communication happens on **intra-node NVLink**, it runs at hundreds of GB/s per GPU, more than an order of magnitude faster than between nodes.

## Recommended configuration: EP=8 + Hybrid-EP

The simplest configuration that also performs well is **8 GPUs per node = one EP group (EP=8, TP=1)**. In this configuration, expert communication (all-to-all) happens only over intra-node NVLink and does not use EFA at all. Between nodes, only the data-parallel communication that synchronizes gradients across nodes holding the same experts goes over EFA.

```
node1: GPU 0~7 = EP group (experts split across 8 GPUs)
node2: GPU 0~7 = EP group (same expert layout as node1)
Between nodes: gradient synchronization across identical experts (data parallelism, EFA)
```

### Build Hybrid-EP as intra-node only

With EP=8, expert communication does not cross a node, so **you can build the Hybrid-EP dispatcher as intra-node only (no multi-node backend).** Inter-node backends (NIXL, DOCA, etc.) are not used.

```bash
export TORCH_CUDA_ARCH_LIST=10.0
git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP
git checkout <specific commit>
DISABLE_AGGRESSIVE_PTX_INSTRS=1 pip install --no-build-isolation --no-deps .
```

- `--no-deps`: prevents pip from reinstalling dependency packages such as PyTorch that are already present in the container image.
- If you do not set `HYBRID_EP_MULTINODE`, it builds as intra-node only. If `Multinode enabled`, `NIXL path`, and `DOCA path` do **not** appear in the build log, it was built correctly as intra-node only.
- **On B200-class hardware, specify `TORCH_CUDA_ARCH_LIST=10.0` directly.** If you do not, it may build with a default for a different architecture.

!!! note "Validated environment (as of 2026-10)"
    - Commit: `94a9f8f6b146c07d97ec58f67cd6d303296d6098`
    - Build time on B200: about 1 minute 30 seconds.
    - Standalone intra-node check (1 node, 8 GPUs): `python tests/test_hybrid_ep.py --num-processes 8` → prints 3 `PASS` results and the bandwidth.

### Megatron dispatcher settings

In the user training code (Megatron-based), specify the Hybrid-EP backend for the flex dispatcher.

```
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep --expert-model-parallel-size 8
```

- To group the EP group within one node, Slurm must place 8 ranks per node contiguously (`--ntasks-per-node=8`). Validation was performed with TP=1, EP=8.
- When using `--moe-grouped-gemm`, `--disable-bias-linear` is required. Without it, model construction fails with `AssertionError: bias_dropout_fusion is not supported in TEGroupedMLP when add_bias_linear=True`.

!!! warning "FP8 dispatch restriction"
    The Hybrid-EP library itself supports FP8 dispatch, but **the `hybridep` dispatcher in Megatron 0.19.0 cannot use FP8 dispatch (a limitation in Megatron code). Use BF16.** If you plan to train in FP8/FP4, verify in advance the precision support scope of the dispatcher and kernels (TransformerEngine, etc.). (This is a version snapshot — it may change in a later version.)

### Expected results

Training should proceed normally — loss should decrease normally with no NaN. In the NCCL log, DP communication is picked up over EFA (`Using network Libfabric`), and because this is an intra-node-only build, inter-node backends such as NIXL and DOCA are not used.

!!! note "Validated environment measurements (2 nodes, EP=8, DP=16)"
    These are results from running a small test model (4 layers, 16 experts) for 30 iterations. **They do not represent the inter-dispatcher performance difference of a real model.**

    - `hybridep` dispatcher: 30 iterations normal, average of the last 10 is **167.8 ms/iteration**
    - `alltoall` dispatcher with the same settings: 184.9 ms/iteration (about 9% slower)

## Example run for validation

This is a small test model configuration to confirm that intra-node Hybrid-EP works correctly at EP=8 (it is not a real training configuration). The launcher fills `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` from Slurm variables.

```bash
#!/bin/bash
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
export MASTER_ADDR=${MASTER_ADDR:?} MASTER_PORT=${MASTER_PORT:-29600}
export CUDA_DEVICE_MAX_CONNECTIONS=1
cd <user training code path>
exec python pretrain_gpt.py \
  --num-layers 4 --hidden-size 2048 --num-attention-heads 16 \
  --disable-bias-linear --swiglu --normalization RMSNorm --position-embedding-type rope --untie-embeddings-and-output-weights \
  --seq-length 4096 --max-position-embeddings 4096 \
  --num-experts 16 --moe-router-topk 2 --moe-ffn-hidden-size 1408 --moe-grouped-gemm --moe-aux-loss-coeff 1e-2 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 8 \
  --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep \
  --micro-batch-size 1 --global-batch-size 64 --train-iters 30 \
  --lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine --lr-warmup-iters 5 --weight-decay 0.1 --clip-grad 1.0 \
  --bf16 --mock-data --tokenizer-type NullTokenizer --vocab-size 32000 \
  --transformer-impl transformer_engine --distributed-backend nccl \
  --log-interval 1 --eval-iters 0 --eval-interval 1000 --save-interval 100000
```

Before full training, it is recommended to run briefly (a few dozen iterations) **with the target model and parallelism configuration as-is** and record the iteration time as a baseline. When you scale up the number of nodes and measure, if the value drops significantly you can compare against this baseline to catch problems early.

## When scaling to EP > 8 (reference)

This does not apply to the recommended configuration (EP=8), but it is reference material for when you later scale EP beyond a node.

### When EP crosses a node in an intra-node-only build

In a Hybrid-EP built as intra-node only, if the EP group crosses a single node (EP > 8, or EP × TP > 8), performance does not simply degrade gradually — **training terminates immediately at startup with the message below.**

```
Inter-node communication is not supported. Please rebuild with HYBRID_EP_MULTINODE flag, group_size=16, buffer_config.num_of_ranks_per_node=8.
```

Remedy: adjust EP ≤ 8, or choose one of the multi-node paths below.

### When inter-node EP is unavoidable

- **The path AWS supports for EFA is DeepEP v2 (the AWS fork, [amazon-contributing/DeepEP](https://github.com/amazon-contributing/DeepEP)).** In the user training code (Megatron-based) you use `--moe-flex-dispatcher-backend deepepv2`. This backend currently exists only in the Megatron `dev` branch, so a stable-version-based fork may require work to incorporate it. The container needs NCCL 2.31 or later and EFA installer 1.50 or later, and the host needs EFA kernel driver 3.3.0 or later (the recommended DLAMI satisfies the host requirement).
- The `deepep` backend (upstream DeepEP) in the stable version (0.19.0) does not work inter-node over EFA.
- If you build Hybrid-EP for multi-node and use it between nodes, performance is low on EFA because it goes through the CPU rather than a direct GPU path.
- With any library, **inter-node all-to-all is capped by EFA bandwidth (about 50 GB/s per GPU)**. So whenever possible, keep EP inside a node and scale out with DP and PP.

!!! warning "Open-source support scope"
    Hybrid-EP (DeepEP) is NVIDIA/DeepSeek open source and **is not covered by AWS support.** DeepEP v2 (the AWS fork) is likewise open source, and the EFA support status of open source changes quickly, so check the latest status right before applying it.

## Hardware memory design reference

B200-class hardware has about 180 GB of memory per GPU (about 1,440 GB per instance). Design TP and PP together with the memory budget so that experts can be split and held across GPU memory at EP=8. Inter-node EFA bandwidth stays at 3.2 Tbps per instance even across GPU generations, so the design principle for all-to-all that crosses nodes (keep EP inside a node) remains unchanged.

## Summary

1. **Keep EP inside a node (≤ 8)**, and scale out between nodes with data parallelism and pipeline parallelism. (EP × TP ≤ 8, with 8 ranks per node placed contiguously.)
2. At EP=8, **build Hybrid-EP as intra-node only** and use the `hybridep` backend as the Megatron dispatcher. FP8 dispatch is unavailable in Megatron 0.19.0 — use BF16.
3. Confirm the dispatcher first with the standalone intra-node test (`test_hybrid_ep.py`), and before full training run briefly with the target configuration as-is to secure the iteration time as a baseline.
4. If you need EP > 8, consider DeepEP v2 (the AWS fork). Hybrid-EP and DeepEP are open source and not covered by AWS support.
