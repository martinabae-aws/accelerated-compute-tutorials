---
title: Region Selection Considerations
description: How region and AZ selection affects AI workloads when building a self-managed large-scale GPU training environment, and the areas to review up front
---

# Region Selection Considerations

When you build large-scale GPU training/inference infrastructure as a **self-managed** environment, region and AZ selection is not a simple location choice — it is a **design decision that affects training performance, cost, and timeline as a whole**. Even for the same instance, the capacity you can secure, storage performance, network configuration, and available services differ by region and AZ.

This page is meant to help you understand "what to watch out for" up front. Because the **specific constraint figures, per-region availability, and remedies for each item vary significantly with the environment**, we recommend confirming the actual design through a discussion with your GTM SSA.

!!! tip "How to use this page"
    The consideration areas below are **checkpoints for "what to review"**. How each item actually manifests as a constraint and how it is resolved depends on the combination of workload, model, timeline, and region — so work through the concrete design and figures together with your GTM SSA (see [How to engage](#how-to-engage) below).

## 1. Define this first — workload requirements

Before discussing regions, the following should be defined. They drive the entire choice of region, instance, and storage.

- **Workload nature**: training (pre-training / fine-tuning / re-training) or inference (online / offline / batch)
- **Model characteristics**: architecture (Dense / MoE / Open-weight), training scope, parameter scale (including active parameters for MoE)
- **Data sovereignty and security**: whether PII is included, whether data can leave the country, specific AZ or redundancy requirements
- **Performance and operational needs**: storage capacity and bandwidth, network performance, DR, MLOps maturity

!!! note "Data sovereignty can decide the region first"
    For regulated industries or sensitive data, data may not be allowed to leave a specific country, which can effectively limit your region options. In that case data sovereignty, rather than performance, becomes the primary constraint. Whether it applies to your case requires a compliance review.

## 2. Considerations differ by build approach

Self-managed can be built in several ways, and the choice changes the region, AZ, and capacity strategy.

- **Amazon EC2** — when you configure it directly or already have your own platform
- **AWS ParallelCluster / PCS** — automates infrastructure management and Slurm scheduling
- **Amazon EKS** — runs training and inference together on Kubernetes
- **Amazon SageMaker HyperPod** — a fully managed platform

Each approach has different per-region support and operational skill requirements. For which approach fits, see the [AI Infrastructure Decision Guide](infra-decision-guide.md), and discuss specific regional suitability with your GTM SSA.

## 3. How region selection affects your AI workload

Here is why region and AZ selection matters, framed by the impact on actual training and inference. **The items below are meant to build awareness that "such constraints may exist"; the specific figures and current availability change frequently, so confirm them with your GTM SSA.**

### Compute — instance generation and availability

- The latest-generation GPU instances are **not offered in every region at the same time.** Whether your desired instance exists in the target region, and whether you can secure the needed quantity, directly affects your timeline.
- Memory, compute performance, and supported data types (FP8/FP4, etc.) differ substantially by generation. **Which instance satisfies the precision and memory your workload requires** is the core of the choice, and because actual performance varies with the model and configuration, a proof of concept (PoC) is needed.

> **Why it matters**: Choosing the wrong instance generation means a model may not fit due to insufficient memory, or conversely you overspend on excess capacity. Generation-to-generation performance differences drive training duration and cost.

### Storage — throughput governs GPU utilization

- High-performance shared storage (e.g., FSx for Lustre) has **availability and performance characteristics that differ by deployment type, region, and AZ.** In some environments throughput may be capped, or a desired performance tier may be available only in certain regions/AZs.
- The storage capacity you can secure may differ by AZ, so capacity should be checked in advance.

> **Why it matters (an often-overlooked point)**: In large-scale training, **GPUs can sit idle waiting for data.** If storage throughput cannot keep up with the rate at which GPUs consume data, expensive GPUs go idle and both training time and cost increase. In other words, a storage constraint is not a "storage" problem but a **"GPU utilization and cost" problem.** That is why you must confirm the availability of storage performance in your region/AZ up front.

### Networking — inter-node communication governs distributed training

- Multi-node distributed training depends on a high-speed inter-node interconnect (EFA), and **the supported EFA generation differs by instance generation.**
- Node placement (physical proximity) affects communication latency, which is tied to the region, AZ, and how capacity is secured.

> **Why it matters**: In distributed training, inter-node communication can become the bottleneck for overall training speed. If communication bandwidth is insufficient or nodes are placed physically far apart, performance does not scale linearly no matter how many GPUs you add, because of communication waits. Network configuration is often fixed at the time of region/AZ selection and is hard to change afterward.

### Operations — MLOps / LLMOps

- Traditional ML and LLM/Agent workloads differ in model size, serving method, monitoring metrics, and cost structure. Operational tooling and platform configuration can be affected by per-region service availability.

## 4. Capacity strategy

When GPU demand is high, a single AZ, single region, or single instance type may not secure the capacity you want. In that case, consider the following.

- **Multi-Region / Multi-AZ**: distribute capacity within what your requirements (data sovereignty, latency) allow
- **Multi-Instance Family**: mixing heterogeneous instances

!!! warning "Mixing heterogeneous instances requires care"
    Mixing different instance generations/types in a single training job can cause unexpected performance degradation or errors due to differences in CPU architecture, network configuration, and storage performance. A common one is leveling-down, where overall performance is capped by the slower side. If mixing is unavoidable, the configuration approach should be discussed with your GTM SSA.

## 5. When using Local Zones

You may consider Local Zones to secure GPU capacity, but there are **differences in supported services and storage performance** compared with the parent region. Some services are not offered directly in a Local Zone, so you may need to design the architecture in conjunction with the parent region.

> **Why it matters**: If a service you assumed would "just work" in a Local Zone is not supported, you may have to roll back the design or add network configuration. Which services are affected should be confirmed in advance.

## 6. When migrating from on-premises

When migrating from an on-premises environment (e.g., InfiniBand-based) to AWS, you should review differences in how networking, storage, and the scheduler behave in advance. In particular, inter-node communication characteristics differ, so performance assumptions from the existing environment may not carry over directly.

## How to engage

The consideration areas on this page are **review checkpoints**. How each item manifests as a constraint on **your specific AI workload, and how to resolve it**, varies significantly with the combination of region, model, timeline, and capacity.

For a thorough review, we recommend **engaging your APJC GTM AI Infra SSA through your account team**. The SSA will help design everything from workload requirements through region, instance, storage, and network configuration.

!!! note "Next step"
    You can review your **GPU training/inference infrastructure region design** together with the AI Infra GTM team through the account team (AM/SA) that handles it. Bringing your workload requirements (section 1 above) prepared in advance will make the discussion move quickly.

    For deeper discussion in the Korea, Japan, and Taiwan region, reach out through your account team to the **APJC GTM AI Infra SSA (Suji Lee)** and **BD (Martina)** for in-depth support.
