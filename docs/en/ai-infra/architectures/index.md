---
title: Reference Architectures
description: Validated architectures for building large-scale AI infrastructure on AWS — Slurm, EKS, and ParallelCluster based references
---

# Reference Architectures

**"Which architecture should you use to build large-scale AI infrastructure on AWS?"** — this section provides validated reference architectures. Each one is a blueprint showing **how to combine** the individual building blocks (DLAMI, Slurm, containers, EFA, GDS, and more).

The implementation details of each building block are documented as reusable components in the [AI Infra Setup](../environment/index.md), [Networking](../efa-setup.md), and [Design](../decisions/index.md) sections. This section ties those blocks together from the perspective of **"here is how this architecture combines them."**

## Architecture catalog

Choose based on your workload scale, operational capability, and orchestration preference.

<div class="grid cards" markdown>

-   :material-server-network:{ .lg .middle } **Slurm-based Large-Scale GPU Training Architecture**

    ---

    Combines Capacity Block + DLAMI + Slurm + containers + EFA + GDS to build a distributed training cluster of tens to hundreds of nodes. The simplest and most proven combination.

    [:octicons-arrow-right-24: View architecture](large-scale-gpu-training.md)

-   :material-kubernetes:{ .lg .middle } **EKS-based** *(coming soon)*

    ---

    An architecture that runs training and inference on a unified Kubernetes platform. Suited to teams with in-house EKS expertise.

-   :material-group:{ .lg .middle } **AWS ParallelCluster-based** *(coming soon)*

    ---

    An architecture that automates infrastructure management and Slurm scheduling with ParallelCluster (or PCS).

</div>

!!! note "More architectures are on the way"

    We currently provide the Slurm-based architecture first. Other architectures — EKS, ParallelCluster, and more — will be added over time. Each reuses the same building blocks (DLAMI, EFA, GDS, and so on); only the orchestration layer differs.
