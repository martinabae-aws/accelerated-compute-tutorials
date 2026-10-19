---
title: Slurm 기반 GPU 학습 클러스터
description: Capacity Block·DLAMI·Slurm·컨테이너·EFA·GDS를 조합한 대규모 분산 학습 클러스터 레퍼런스 아키텍처
---

# Slurm 기반 GPU 학습 클러스터

이 아키텍처는 **수십~수백 노드 규모의 GPU 분산 학습 클러스터** 를 Slurm 기반으로 구성합니다. 각 단계는 해당 기술 요소의 전용 가이드로 연결되며, 이 페이지는 그 **조합 순서와 연결 관계** 를 보여주는 청사진입니다.

## 아키텍처 개요

```
[ Capacity Block (GPU 노드 × N) ]
  각 노드: DLAMI Base (NVIDIA 드라이버·EFA·GDS·Lustre client 사전 설치)
    ├─ 호스트: Slurm(slurmd), munge, Enroot, Pyxis 추가
    └─ 학습 잡: srun → Pyxis → Enroot 컨테이너(NGC + EFA + 학습 코드)
  네트워크: 노드 안 NVLink / 노드 간 EFA × 8
  공유 스토리지: FSx for Lustre (GDS over EFA)
```

이 조합이 **가장 단순하고 검증된** 대규모 학습 구성입니다. 오케스트레이션 계층을 최소화(Slurm만)하여, 분산 학습에 집중할 수 있습니다.

## 구성 순서

아래 순서대로 진행하십시오. 각 단계는 전용 가이드에 상세 명령·검증 방법이 있습니다.

### 1단계 — 용량 확보: Capacity Block

대규모 GPU는 [Capacity Blocks for ML](../capacity-blocks-guide.md)로 확보합니다. 한 블록 안의 노드는 저지연 네트워크로 가깝게 배치됩니다.

### 2단계 — 호스트 구성: DLAMI

각 노드의 호스트를 [DLAMI 호스트 구성·검증](../environment/dlami-host-setup.md) 가이드에 따라 준비합니다. 드라이버·EFA·GDS·Lustre client가 사전 설치된 DLAMI Base를 쓰고, **커널을 고정** 하는 것이 핵심입니다.

### 3단계 — 스케줄러: Slurm + Pyxis + Enroot

[Slurm + Pyxis + Enroot](../environment/slurm-pyxis-enroot.md) 가이드로 스케줄러와 컨테이너 런타임을 구성합니다. 잡 제출 시 `--cpus-per-task`·`--export=ALL`이 성능·동작의 핵심입니다.

### 4단계 — 학습 컨테이너: NGC + EFA

[컨테이너 이미지 (NGC + EFA)](../environment/container-ngc-efa.md) 가이드로 학습 이미지를 만듭니다. NGC 이미지에 EFA installer를 더하고, 학습 코드를 올립니다.

### 5단계 — 네트워크 심화: EFA

[EFA 네트워크 심화](../efa-setup.md)에서 노드 간 통신(all-reduce·all-to-all)의 원리와 NCCL 플러그인 검증을 다룹니다. 대규모 학습의 성능을 좌우하는 핵심입니다.

### 6단계 — 공유 스토리지: GDS

[GDS (GPUDirect Storage)](../environment/gds-gpudirect-storage.md)로 FSx for Lustre를 GPU 메모리에 직접 연결(GDS over EFA)해 데이터 로딩 병목을 줄입니다.

### 7단계 — 검증: 분산 학습 go/no-go

[분산 학습 검증 (nccl-tests)](../environment/slurm-pyxis-enroot.md#distributed-training-validation)으로 본 학습 전에 성능 기준선을 측정합니다. go/no-go 게이트를 통과해야 다음으로 갑니다.

### 8단계 (선택) — MoE 학습: Hybrid-EP

MoE 모델을 학습한다면 [MoE / Hybrid-EP 검증](../environment/moe-hybrid-ep.md)에서 Expert Parallel 설계(EP는 노드 안에)와 Hybrid-EP 빌드를 확인합니다.

## 설계 참고

구성에 앞서 "어느 리전에, 어떤 오케스트레이터로 세울지"는 설계 단계에서 결정합니다.

- [AI 인프라 의사결정 가이드](../infra-decision-guide.md) — EC2 vs ParallelCluster vs EKS 선택
- [리전 선택 고려사항](../self-hosted-llm-considerations.md) — 리전·서비스 GA·용량 전략
- [가속기 선택 가이드](../xc-selection-guide.md) — 어떤 GPU 인스턴스를 쓸지

!!! tip "다른 아키텍처"

    이 아키텍처는 Slurm 기반입니다. EKS 기반, ParallelCluster 기반 아키텍처는 [실전 아키텍처](index.md) 허브에서 순차 추가될 예정입니다. 동일한 기술 요소 블록(DLAMI·EFA·GDS 등)을 재사용하며, 오케스트레이션 계층만 달라집니다.
