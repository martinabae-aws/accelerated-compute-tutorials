---
title: 실전 아키텍처
description: AWS에서 대규모 AI 인프라를 구성하는 검증된 아키텍처 — Slurm, EKS, ParallelCluster 기반 레퍼런스
---

# 실전 아키텍처

**"AWS에서 대규모 AI 인프라를 어떤 아키텍처로 구성하는가?"** — 이 섹션은 검증된 레퍼런스 아키텍처를 제공합니다. 각 아키텍처는 개별 기술 요소(DLAMI, Slurm, 컨테이너, EFA, GDS 등)를 **어떻게 조합하는지** 를 보여주는 청사진입니다.

각 기술 요소의 세부 구현은 [AI 인프라 구축](../environment/index.md)·[네트워킹](../efa-setup.md)·[설계](../decisions/index.md) 섹션에 재사용 가능한 블록으로 정리되어 있습니다. 이 섹션은 그 블록들을 **"이 아키텍처에서는 이렇게 조합한다"** 는 관점으로 엮습니다.

## 아키텍처 목록

워크로드 규모·운영 역량·오케스트레이션 선호에 따라 선택하십시오.

<div class="grid cards" markdown>

-   :material-server-network:{ .lg .middle } **Slurm 기반 대규모 GPU 학습 아키텍처**

    ---

    Capacity Block + DLAMI + Slurm + 컨테이너 + EFA + GDS를 조합해 수십~수백 노드 규모의 분산 학습 클러스터를 구성합니다. 가장 단순하고 검증된 조합입니다.

    [:octicons-arrow-right-24: 아키텍처 보기](large-scale-gpu-training.md)

-   :material-kubernetes:{ .lg .middle } **EKS 기반** *(준비 중)*

    ---

    Kubernetes로 학습·추론을 통합 운영하는 아키텍처. ML 팀이 EKS 역량을 보유한 경우에 적합합니다.

-   :material-group:{ .lg .middle } **AWS ParallelCluster 기반** *(준비 중)*

    ---

    ParallelCluster(또는 PCS)로 인프라 관리와 Slurm 스케줄링을 자동화하는 아키텍처.

</div>

!!! note "아키텍처는 계속 추가됩니다"

    현재는 Slurm 기반 아키텍처를 먼저 제공합니다. EKS, ParallelCluster 등 다른 아키텍처를 순차적으로 추가할 예정입니다. 각 아키텍처는 동일한 기술 요소 블록(DLAMI·EFA·GDS 등)을 재사용하며, 오케스트레이션 계층만 달라집니다.
