---
title: Slurm-based GPU Training Cluster
description: A reference architecture for a large-scale distributed training cluster combining Capacity Block, DLAMI, Slurm, containers, EFA, and GDS
---

# Slurm-based GPU Training Cluster

This architecture builds a **GPU distributed training cluster of tens to hundreds of nodes** on Slurm. Each step links to the dedicated guide for the corresponding building block; this page is the blueprint that shows **the order in which they combine and how they connect**.

## Architecture overview

```
[ Capacity Block (GPU nodes × N) ]
  Each node: DLAMI Base (NVIDIA driver, EFA, GDS, Lustre client pre-installed)
    ├─ Host: add Slurm (slurmd), munge, Enroot, Pyxis
    └─ Training job: srun → Pyxis → Enroot container (NGC + EFA + training code)
  Network: intra-node NVLink / inter-node EFA × 8
  Shared storage: FSx for Lustre (GDS over EFA)
```

This combination is the **simplest and most proven** large-scale training setup. By keeping the orchestration layer minimal (Slurm only), you can focus on distributed training.

## Build sequence

Proceed in the order below. Each step has detailed commands and validation methods in its dedicated guide.

### Step 1 — Secure capacity: Capacity Block

Secure large-scale GPUs with [Capacity Blocks for ML](../capacity-blocks-guide.md). Nodes within a single block are placed close together on a low-latency network.

### Step 2 — Host setup: DLAMI

Prepare each node's host following the [DLAMI Host Setup & Validation](../environment/dlami-host-setup.md) guide. Use the DLAMI Base with the driver, EFA, GDS, and Lustre client pre-installed, and **pin the kernel** — this is the key point.

### Step 3 — Scheduler: Slurm + Pyxis + Enroot

Configure the scheduler and container runtime with the [Slurm + Pyxis + Enroot](../environment/slurm-pyxis-enroot.md) guide. When submitting jobs, `--cpus-per-task` and `--export=ALL` are critical for performance and correct behavior.

### Step 4 — Training container: NGC + EFA

Build the training image with the [Container Image (NGC + EFA)](../environment/container-ngc-efa.md) guide. Add the EFA installer to the NGC image and layer your training code on top.

### Step 5 — Networking deep dive: EFA

[EFA Networking Deep Dive](../efa-setup.md) covers the principles of inter-node communication (all-reduce, all-to-all) and NCCL plugin validation. This is what determines large-scale training performance.

### Step 6 — Shared storage: GDS

Use [GDS (GPUDirect Storage)](../environment/gds-gpudirect-storage.md) to connect FSx for Lustre directly to GPU memory (GDS over EFA) and reduce the data-loading bottleneck.

### Step 7 — Validation: distributed training go/no-go

Measure a performance baseline before full training with [Distributed Training Validation (nccl-tests)](../environment/slurm-pyxis-enroot.md#distributed-training-validation). You must pass the go/no-go gate before moving on.

### Step 8 (optional) — MoE training: Hybrid-EP

If you train MoE models, review the Expert Parallel design (keep EP within a node) and the Hybrid-EP build in [MoE / Hybrid-EP](../environment/moe-hybrid-ep.md).

## Design references

Before building, decide "which region and which orchestrator to use" at the design stage.

- [AI Infrastructure Decision Guide](../infra-decision-guide.md) — choosing between EC2, ParallelCluster, and EKS
- [Region Selection Considerations](../self-hosted-llm-considerations.md) — region, service GA, and capacity strategy
- [Accelerator Selection Guide](../xc-selection-guide.md) — which GPU instance to use

!!! tip "Other architectures"

    This architecture is Slurm-based. EKS-based and ParallelCluster-based architectures will be added over time in the [Reference Architectures](index.md) hub. Each reuses the same building blocks (DLAMI, EFA, GDS, and so on); only the orchestration layer differs.
