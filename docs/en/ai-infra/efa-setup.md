---
title: EFA (Elastic Fabric Adapter)
description: Understanding the bandwidth gap between intra-node NVLink and inter-node EFA in Elastic Fabric Adapter (EFA)-based multi-node GPU training, and how to correctly design all-reduce / all-to-all communication, validate the NCCL plugin, and place nodes
---

# EFA (Elastic Fabric Adapter)

This page covers how to understand where the network determines training performance in multi-node GPU training environments connected by Elastic Fabric Adapter (EFA), and how to design and validate them correctly. It is especially relevant for workloads sensitive to inter-node communication, such as Mixture-of-Experts (MoE) models. The body explains principles, sequence, and architecture that remain valid regardless of version, while point-in-time measurements, versions, and commands are separated into "Validated environment" boxes.

## Core principles

The network design for large-scale distributed training can be summarized in the following four points.

1. **Within a node (8 GPUs) it is NVLink; between nodes it is EFA.** On the latest B200-generation GPU instances, EFA provides 3.2 Tbps (400 GB/s) per instance, which divided across 8 GPUs is **about 50 GB/s per GPU**. By contrast, intra-node NVLink is about 900 GB/s per GPU in one direction, making the bandwidth gap between the two paths more than tenfold. This asymmetry is the starting point for all communication design.

2. **all-reduce (data parallelism) uses nearly all of the EFA bandwidth.** Because all-reduce distributes communication volume with ring/tree algorithms, performance approaches the EFA bandwidth even when the inter-node path is saturated. In other words, data-parallel scaling operates efficiently over EFA.

3. **When all-to-all (expert parallelism) crosses nodes, the per-GPU EFA share (about 50 GB/s) becomes the ceiling.** Because all-to-all has every rank exchanging data with every other rank, it is much more sensitive to bandwidth and latency. The moment it crosses nodes, the per-GPU inter-node traffic converges to the EFA share.

4. **Therefore, in MoE the most important design is to keep expert parallelism (EP) within a node (≤ 8).** When an EP group fits within a single node, expert communication occurs only over NVLink at hundreds of GB/s, and only the data-parallel all-reduce travels between nodes over EFA.

!!! note "Validated environment (as of 2026-10)"
    The figures below are reference values measured in a 2-node (16 GPU) environment of the latest B200-generation GPU instances, aligned to the main software versions (NGC container image, MoE communication libraries, Megatron version, etc.). They do not guarantee performance and vary with region, node count, controller configuration, and training code. Measure a baseline in your actual environment using the same method (this page's "Recommended pre-validation sequence") and compare.

    - all-reduce (16 GiB, nccl-tests): algbw **374 GB/s** — about 94% of the per-node EFA 400 GB/s (computed value)
    - all-to-all (16 GiB, nccl-tests): algbw **95 GB/s**, per-GPU inter-node traffic converted to about 48 GB/s — about 95% of the per-GPU EFA share of 50 GB/s (computed value)

## MoE all-to-all (Expert Parallelism)

### Mechanism

MoE performs **all-to-all** communication to route tokens to experts and bring them back. Because every rank exchanges data with every other rank, it is much more sensitive to network bandwidth and latency than data-parallel all-reduce.

**When the EP degree exceeds the number of GPUs in a node (8)**, all-to-all traffic leaves NVLink and goes out over EFA. From this point on, EP communication performance is bounded by EFA bandwidth. Conversely, when the EP group fits within a single node, expert communication is handled by NVLink bandwidth (hundreds of GB/s).

### Performance comparison by communication path

| Communication | Path | Measured value | Meaning |
|---|---|---|---|
| all-reduce (16 GiB) | Intra-node NVLink + inter-node EFA (NCCL NVLS tree) | algbw 374 GB/s | About 94% of per-node EFA 400 GB/s (computed value) |
| all-to-all (16 GiB) | Inter-node EFA | algbw 95 GB/s (per-GPU inter-node traffic about 48 GB/s) | About 95% of per-GPU EFA share of 50 GB/s (computed value) |
| Intra-node EP dispatch / combine (8 GPU, hidden 7168, 4096 tokens/rank, top-8) | Intra-node NVLink | API-based dispatch 419 GB/s (FP8), combine 512 GB/s | Several times to about 10x the per-GPU inter-node all-to-all (measurement method differs, so treat the direct comparison as a reference) |

The key conclusion is that **EFA itself is already being used nearly to its limit, so inter-node all-to-all performance is determined by design (whether EP is kept within a node) rather than by tuning**. The ceiling for inter-node all-to-all is about 50 GB/s per GPU, and no library can exceed this wall.

### Recommended configuration: keep EP within a node

If you set EP to be at most the number of GPUs in a node (e.g., EP=8, TP=1, PP=1), the 8 GPUs of each node form one EP group, so expert communication occurs only over intra-node NVLink and does not use EFA. Between nodes, only the data-parallel communication that reconciles gradients across nodes holding the same experts travels over EFA.

For an EP group to fit within a single node, you must have **EP × TP ≤ 8** and place 8 ranks per node contiguously. When this condition is met, training works correctly even with an intra-node-only build (without an inter-node backend). Data-parallel communication operates over EFA (`Using network Libfabric`); verify that loss decreases normally and that no NaN occurs.

!!! warning "If the EP group crosses nodes, it may terminate at startup"
    With an intra-node-only build, if the EP group crosses a node (EP > 8, or EP × TP > 8), training terminates immediately at startup with a message like the following. Rather than degrading quietly, it tells you at startup, so read the message and fix the parallelization configuration.

    ```
    Inter-node communication is not supported. Please rebuild with
    HYBRID_EP_MULTINODE flag, group_size=16, buffer_config.num_of_ranks_per_node=8.
    ```

### When scaling to EP > 8 (reference)

If you must scale EP beyond a node, consider the following. Regardless of which path you choose, remember that **inter-node all-to-all is bounded by EFA bandwidth (about 50 GB/s per GPU)**.

- **The path supported for EFA is DeepEP v2 (AWS fork, [amazon-contributing/DeepEP](https://github.com/amazon-contributing/DeepEP)).** This backend requires NCCL 2.31 or later, EFA installer 1.50 or later (container), and the host's EFA kernel driver 3.3.0 or later. The recommended DLAMI satisfies the host requirements.
- The upstream DeepEP backend does not operate across nodes on EFA.
- If you build an intra-node-only EP library for multi-node and use it across nodes, it does not use a GPU-direct path on EFA but goes through the CPU, so performance is low.
- When scaling, keep the target model and parallelization configuration as-is, increase the node count, and measure to compare.

!!! note "Validated environment (as of 2026-10)"
    The EP=8 configuration above was confirmed to work with the following version combination. This combination is a point-in-time snapshot; when versions change, you only need to update this box.

    - Container image: `nvcr.io/nvidia/pytorch:26.08-py3` (+ EFA installer 1.50.0)
    - Intra-node-only EP library: specific commit `94a9f8f6b146c07d97ec58f67cd6d303296d6098` (intra-node-only build)
    - Megatron-LM 0.19.0, `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep`
    - Parallelization: EP=8, TP=1, PP=1, 2 nodes 16 GPUs
    - The DeepEP v2 (AWS fork) backend is only in the Megatron `dev` branch, so a 0.19.0-based fork requires integration work (`--moe-flex-dispatcher-backend deepepv2`).
    - Reference comparison: with the same settings, iteration time was about 9% shorter versus the NCCL `alltoall` dispatcher (based on a small test model).

## Node placement (Capacity Block · spine)

Inter-node communication performance is affected by where instances are placed in the network topology.

- **Instances within a single Capacity Block are automatically placed close together within an EC2 UltraCluster and connected over a low-latency non-blocking network** ([Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)). If you reserve N nodes as a single Capacity Block, AWS handles placement within the block.
- Large-scale (e.g., tens of nodes) Capacity Blocks are not offered in every region or instance type. Before purchasing, verify that the desired block size for that instance type is available in `<region>`.
- If you reserve a Capacity Block split into several, the blocks may be placed in different network segments (spines) and cannot be merged afterward. In this case, inter-block communication performance may differ from intra-block, so verify the placement and measure.
- **You cannot specify a Placement Group for a Capacity Block (not supported).** Do not include a Placement Group in the launch template or launch command ([Capacity Blocks — Considerations](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html#capacity-blocks-considerations)).

### Checking placement and measuring between blocks

```bash
aws ec2 describe-instance-topology --instance-ids <id> ...
```

By comparing the `NetworkNodes` hierarchy in the result from the bottom (the side closer to the instance), you can tell which instances share the same spine. If you use multiple blocks, measure `alltoall` and `fi_pingpong` for an intra-block node pair and an inter-block node pair respectively to check the performance difference (see "Recommended pre-validation sequence" below).

## Common considerations when migrating InfiniBand → EFA

If settings used in an IB environment remain in the EFA environment, performance can drop significantly or initialization can fail. When migrating, check the following.

### Always validate that NCCL uses EFA

If the plugin is not picked up, NCCL falls back to Socket (TCP) and performance drops significantly. Always confirm the following two lines in the `NCCL_DEBUG=INFO` log.

- `Using network Libfabric`
- `Selected provider is efa ... (found 8 nics)`

### Remove leftover IB settings

The latest NGC images default to the OFI plugin, so the EFA plugin is selected automatically. However, if IB settings (`NCCL_NET=IB`, changing `NCCL_NET_PLUGIN`, `NCCL_IB_*`) remain in the EFA job script, EFA may not be used (specifying `NCCL_NET=IB` causes initialization to fail). In an EFA environment, remove all IB-oriented `NCCL_*` settings. `NCCL_NET` does not need to be specified, and if you insist on specifying it, it is `Libfabric`.

### EFA configuration in the container

The latest NGC images embed the EFA stack (libfabric, aws-ofi-nccl), so they deliver normal performance as-is. In Pyxis/Enroot environments, `/dev/infiniband` is visible without a separate mount. However, for the training image we recommend installing EFA installer 1.50 or later to match the host versions and to prepare for inter-node backends such as DeepEP v2.

### Caution about ConnectX-7 device confusion

On the latest B200-generation instances, two ConnectX-7 devices with no link connected may be visible besides EFA. Since NCCL selects EFA, there is no impact. Do not see these devices and force the IB path — that leads straight to the pitfall of "Remove leftover IB settings" just above.

### GPUDirect RDMA

Direct inter-node GPU memory transfer (GPUDirect RDMA) is enabled in the default configuration. Confirm `via NET/Libfabric/.../GDRDMA` in the NCCL log. If this path is off, inter-node transfer goes through host memory and slows down.

### Per-rank CPU allocation

If only 1 CPU core is assigned per rank, communication performance can drop significantly even when EFA is healthy. Each rank needs enough cores — the node's vCPU count divided by the number of GPUs. How to specify this with your scheduler (for example, Slurm's `--cpus-per-task`) is covered in the [Submitting jobs](environment/slurm-pyxis-enroot.md) section.

!!! note "Validated environment (as of 2026-10)"
    - Impact of CPU under-allocation: launching without `--cpus-per-task` dropped 8 GiB all-reduce algbw from 368 to 92 GB/s and all-to-all from 95 to 8.7 GB/s — a drop of 1/4 to 1/10.
    - Recommendation: `--cpus-per-task=24` (based on 192 vCPU ÷ 8 GPU).
    - If the version combination (driver, CUDA, EFA, aws-ofi-nccl, NCCL) is mismatched, performance degradation and errors occur, so use a validated DLAMI and avoid arbitrary kernel upgrades.

## Target hardware specifications and notes

The representative specifications of the latest B200-generation GPU instances are as follows.

!!! note "Validated environment (as of 2026-10) — P6-B200 specifications"
    | Item | Value |
    |---|---|
    | GPU | NVIDIA B200 × 8 (about 180 GB memory per GPU, about 1,440 GB per instance) |
    | GPU interconnect | NVLink (NV18 between GPUs within a node) |
    | EFA | 8 network cards, 3,200 Gbps per instance (about 50 GB/s per GPU) |
    | vCPU / memory | 192 vCPU / about 2 TB |
    | Local NVMe | 3.84 TB × 8 (DLAMI auto-mounts them bundled at `/opt/dlami/nvme`, about 28 TB) |

Notes that remain valid regardless of version are as follows.

- **EFA bandwidth is the same 3.2 Tbps per instance as the previous generation (H200).** Even as the GPU generation changes, the inter-node communication bandwidth stays the same, so the design principle for cross-node all-to-all (keep EP within a node) remains valid.
- Design TP, PP, and the memory budget so that EP can be split and held within the number of nodes inside GPU memory.
- When using FP8/FP4, check in advance the precision/convergence and the kernel (TransformerEngine, etc.) support scope. The intra-node EP library itself supports FP8 dispatch, but FP8 dispatch cannot be used with the `hybridep` dispatcher in Megatron 0.19.0 (a limitation in the Megatron code).

## Recommended pre-validation sequence

Before full training, confirm in the following order that the network and communication paths behave as designed, and establish a baseline (iteration time) for the target configuration. **The concrete execution commands for each step (nccl-tests, `fi_pingpong`, container measurement) are collected in [Distributed training validation](environment/slurm-pyxis-enroot.md#distributed-training-validation).** Here we only summarize what to check and why.

1. **Host EFA/NCCL baseline** — measure all-reduce and all-to-all bandwidth over 2+ nodes and confirm the EFA path (`Using network Libfabric`) approaches the design bandwidth.
2. **Container NCCL path** — confirm the training image delivers performance similar to the host, since the image's EFA stack and plugin settings can differ from the host.
3. **Inter-node latency** — establish a baseline for latency-sensitive communication with small-message round-trip latency (`fi_pingpong`).
4. **Verify the EP dispatcher** — for MoE, verify the Expert Parallelism backend standalone ([MoE / Hybrid-EP](environment/moe-hybrid-ep.md)).
5. **Short training with the target model and parallelization configuration** — record iteration time as the baseline.
6. (If there are multiple Capacity Blocks) after checking the topology, compare intra-block and inter-block.

## References — official and primary sources

**EFA · NCCL**

- [Get started with EFA and NCCL for ML workloads on Amazon EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html) · [EFA overview](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [Maximize network bandwidth — EFA configuration for P6-B200](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-acc-inst-types.html#efa-for-p6-b200)
- [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) · [libfabric EFA provider (`fi_efa.7`)](https://github.com/ofiwg/libfabric/blob/main/man/fi_efa.7.md) · [`fi_pingpong(1)`](https://ofiwg.github.io/libfabric/main/man/fi_pingpong.1.html)
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) · [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) · [nccl-tests](https://github.com/NVIDIA/nccl-tests)

**Capacity Block · placement**

- [Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html) · [Launch instances using Capacity Blocks](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-launch.html)
- [EC2 instance topology](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-topology.html) · [describe-instance-topology](https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instance-topology.html)

**MoE EP communication**

- [DeepEP](https://github.com/deepseek-ai/DeepEP) · [DeepEP v2 — AWS fork](https://github.com/amazon-contributing/DeepEP) · [deepep-v2-benchmark](https://github.com/awslabs/awsome-distributed-ai/tree/main/micro-benchmarks/expert-parallelism/deepep-v2-benchmark)
- [Megatron-LM PR #4793 (DeepEP v2 backend, dev)](https://github.com/NVIDIA/Megatron-LM/pull/4793) · [PR #5153 (merged to main)](https://github.com/NVIDIA/Megatron-LM/pull/5153)
- [NVIDIA/nccl #2272 (NCCL roadmap: EFA GDA support in GIN)](https://github.com/NVIDIA/nccl/issues/2272)

!!! note "Note on sources"
    GitHub and NVIDIA documents are primary sources, not official AWS documentation. The EFA support status of open source changes quickly, so check the latest status right before applying.
