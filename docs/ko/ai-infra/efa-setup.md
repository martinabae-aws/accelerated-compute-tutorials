---
title: EFA (Elastic Fabric Adapter)
description: Elastic Fabric Adapter(EFA) 기반 다중 노드 GPU 학습에서 노드 내 NVLink와 노드 간 EFA의 대역폭 차이를 이해하고, all-reduce·all-to-all 통신, NCCL 플러그인 검증, 노드 배치를 올바르게 설계하는 방법
---

# EFA (Elastic Fabric Adapter)

이 페이지는 Elastic Fabric Adapter(EFA)로 연결된 다중 노드 GPU 학습 환경에서, 네트워크가 학습 성능을 좌우하는 지점을 이해하고 올바르게 설계·검증하는 방법을 다룹니다. 특히 전문가 혼합(MoE) 모델처럼 노드 간 통신에 민감한 워크로드를 염두에 둡니다. 본문은 버전과 무관하게 유효한 원칙·순서·아키텍처를 설명하고, 특정 시점의 측정값·버전·명령은 `검증 환경` 박스로 분리했습니다.

## 핵심 원칙

대규모 분산 학습의 네트워크 설계는 다음 네 가지로 요약됩니다.

1. **노드 안(GPU 8개)은 NVLink, 노드 사이는 EFA다.** EFA는 최신 B200 세대 GPU 인스턴스에서 인스턴스당 3.2 Tbps(400 GB/s)이며, GPU 8개로 나누면 **GPU당 약 50 GB/s** 입니다. 반면 노드 안 NVLink는 GPU당 한 방향 약 900 GB/s로, 두 경로의 대역폭 차이가 10배 이상입니다. 이 비대칭이 모든 통신 설계의 출발점입니다.

2. **all-reduce(데이터 병렬)는 EFA 대역폭을 거의 다 쓴다.** all-reduce는 링/트리 알고리즘으로 통신량을 분산하므로, 노드 간 경로가 포화되더라도 EFA 대역폭에 근접한 성능이 나옵니다. 즉 데이터 병렬 확장은 EFA 위에서 효율적으로 동작합니다.

3. **all-to-all(Expert 병렬)이 노드를 넘으면 GPU당 EFA 몫(약 50 GB/s)이 상한이 된다.** all-to-all은 모든 rank가 모든 rank와 데이터를 주고받으므로 대역폭과 지연에 훨씬 민감합니다. 노드를 넘는 순간 GPU당 노드 간 트래픽은 EFA 몫에 수렴합니다.

4. **그래서 MoE에서는 Expert 병렬(EP)을 노드 안(≤ 8)에 두는 설계가 가장 중요하다.** EP 그룹이 한 노드 안에 들어가면 expert 통신이 NVLink에서만 일어나 수백 GB/s로 동작하고, 노드 사이에는 데이터 병렬의 all-reduce만 EFA로 오갑니다.

!!! note "검증 환경 (2026-10 기준)"
    아래 수치는 주요 소프트웨어 버전(NGC 컨테이너 이미지, MoE 통신 라이브러리, Megatron 버전 등)을 맞춘 최신 B200 세대 GPU 인스턴스 2노드(16 GPU) 환경에서 측정한 참고값입니다. 성능을 보장하는 값이 아니며, 리전·노드 수·컨트롤러 구성·학습 코드에 따라 달라집니다. 실제 환경에서 같은 방법(이 페이지의 "권장 사전 검증 순서")으로 기준선을 측정해 비교하십시오.

    - all-reduce(16 GiB, nccl-tests): algbw **374 GB/s** — 노드당 EFA 400 GB/s의 약 94% 수준(계산값)
    - all-to-all(16 GiB, nccl-tests): algbw **95 GB/s**, GPU당 노드 간 트래픽 환산 약 48 GB/s — GPU당 EFA 몫 50 GB/s의 약 95% 수준(계산값)

## MoE all-to-all (Expert Parallelism)

### 메커니즘

MoE는 토큰을 expert로 보내고 돌려받기 위해 **all-to-all** 통신을 씁니다. 모든 rank가 모든 rank와 데이터를 주고받으므로, 데이터 병렬의 all-reduce보다 네트워크 대역폭과 지연에 훨씬 민감합니다.

**EP degree가 노드의 GPU 수(8)를 넘으면** all-to-all 트래픽이 NVLink를 벗어나 EFA로 나갑니다. 이때부터 EP 통신 성능은 EFA 대역폭이 정합니다. 반대로 EP 그룹이 한 노드 안에 들어가면 expert 통신은 NVLink 대역폭(수백 GB/s)으로 처리됩니다.

### 통신 경로별 성능 비교

| 통신 | 경로 | 측정값 | 의미 |
|---|---|---|---|
| all-reduce (16 GiB) | 노드 안 NVLink + 노드 간 EFA (NCCL NVLS tree) | algbw 374 GB/s | 노드당 EFA 400 GB/s의 약 94%(계산값) |
| all-to-all (16 GiB) | 노드 간 EFA | algbw 95 GB/s (GPU당 노드 간 트래픽 약 48 GB/s) | GPU당 EFA 몫 50 GB/s의 약 95%(계산값) |
| 노드 안 EP dispatch / combine (8 GPU, hidden 7168, 4096 토큰/rank, top-8) | 노드 안 NVLink | API 기준 dispatch 419 GB/s(FP8), combine 512 GB/s | GPU당 노드 간 all-to-all보다 수 배~약 10배 수준(측정 방식이 달라 단순 비교는 참고용) |

핵심 결론은 **EFA 자체는 이미 거의 한계까지 쓰이고 있으므로, 노드 간 all-to-all 성능은 튜닝보다 설계(EP를 노드 안에 두는가)가 좌우한다** 는 점입니다. 노드 간 all-to-all의 상한은 GPU당 약 50 GB/s이며, 어떤 라이브러리를 쓰더라도 이 벽을 넘을 수 없습니다.

### 권장 구성: EP를 노드 안에 두기

EP를 노드의 GPU 수 이하로 설정하면(예: EP=8, TP=1, PP=1), 각 노드의 GPU 8개가 하나의 EP 그룹이 되어 expert 통신이 노드 안 NVLink에서만 일어나고 EFA를 쓰지 않습니다. 노드 사이에는 같은 expert를 가진 노드끼리 gradient를 맞추는 데이터 병렬 통신만 EFA로 오갑니다.

EP 그룹이 한 노드 안에 들어가려면 **EP × TP ≤ 8** 이어야 하고, 노드당 8 rank를 연속 배치해야 합니다. 이 조건을 만족하면 노드 내 전용 빌드(노드 간 백엔드 없이)로도 학습이 정상 동작합니다. 데이터 병렬 통신은 EFA(`Using network Libfabric`)로 동작하며, loss가 정상 감소하고 NaN이 발생하지 않는지로 확인합니다.

!!! warning "EP 그룹이 노드를 넘으면 시작 시점에 종료될 수 있습니다"
    노드 내 전용 빌드에서 EP 그룹이 한 노드를 넘으면(EP > 8, 또는 EP × TP > 8), 학습 시작 시점에 다음과 같은 메시지와 함께 바로 종료됩니다. 성능이 조용히 떨어지는 것이 아니라 시작할 때 알려 주므로, 메시지를 보고 병렬화 구성을 바로잡으십시오.

    ```
    Inter-node communication is not supported. Please rebuild with
    HYBRID_EP_MULTINODE flag, group_size=16, buffer_config.num_of_ranks_per_node=8.
    ```

### EP > 8로 확장할 경우 (참고)

EP를 노드 밖으로 확장해야 한다면 다음을 고려합니다. 어떤 경로를 택하든 **노드 간 all-to-all은 EFA 대역폭(GPU당 약 50 GB/s)이 상한** 임을 기억하십시오.

- **EFA용으로 지원되는 경로는 DeepEP v2(AWS 포크, [amazon-contributing/DeepEP](https://github.com/amazon-contributing/DeepEP))** 입니다. 이 백엔드는 NCCL 2.31 이상, EFA installer 1.50 이상(컨테이너), 호스트의 EFA 커널 드라이버 3.3.0 이상을 요구합니다. 권장 DLAMI는 호스트 조건을 충족합니다.
- 원본 DeepEP 백엔드는 EFA에서 노드 간으로 동작하지 않습니다.
- 노드 내 전용 EP 라이브러리를 멀티노드로 빌드해 노드 간에 쓰면, EFA에서는 GPU 직접 경로가 아니라 CPU를 거치는 방식이라 성능이 낮습니다.
- 확장 시에는 대상 모델·병렬화 구성 그대로 노드 수를 늘려 가며 실측해 비교하십시오.

!!! note "검증 환경 (2026-10 기준)"
    위 EP=8 구성은 다음 버전 조합에서 동작을 확인했습니다. 이 조합은 특정 시점 스냅샷이며, 버전이 바뀌면 이 박스만 갱신하면 됩니다.

    - 컨테이너 이미지: `nvcr.io/nvidia/pytorch:26.08-py3` (+ EFA installer 1.50.0)
    - 노드 내 전용 EP 라이브러리: 특정 커밋 `94a9f8f6b146c07d97ec58f67cd6d303296d6098` (노드 내 전용 빌드)
    - Megatron-LM 0.19.0, `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep`
    - 병렬화: EP=8, TP=1, PP=1, 2노드 16 GPU
    - DeepEP v2(AWS 포크) 백엔드는 Megatron `dev` 브랜치에만 있어 0.19.0 기반 포크에는 반영 작업이 필요합니다(`--moe-flex-dispatcher-backend deepepv2`).
    - 참고 비교: 같은 설정에서 NCCL `alltoall` 디스패처 대비 iteration 시간 약 9% 짧음(작은 시험 모델 기준).

## 노드 배치 (Capacity Block · spine)

노드 간 통신 성능은 인스턴스가 네트워크 토폴로지상 어디에 배치되는가에 영향을 받습니다.

- **한 Capacity Block 안의 인스턴스는 EC2 UltraCluster 안에 자동으로 가깝게 배치되며, 저지연 non-blocking 네트워크로 연결됩니다**([Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)). N개의 노드를 하나의 Capacity Block으로 확보하면 블록 안의 배치는 AWS가 처리합니다.
- 대규모(예: 수십 대) Capacity Block은 모든 리전·인스턴스 타입에서 제공되지는 않습니다. 구매 전에 `<region>`에서 해당 인스턴스 타입의 원하는 블록 크기가 조회되는지 확인하십시오.
- Capacity Block을 여러 개로 나눠 확보하면 블록끼리 서로 다른 네트워크 구간(spine)에 배치될 수 있고, 사후에 합칠 수 없습니다. 이 경우 블록 사이 통신 성능이 블록 안과 다를 수 있으므로, 배치를 확인하고 측정하십시오.
- **Capacity Block에는 Placement Group을 지정할 수 없습니다(미지원).** 런치 템플릿이나 기동 명령에 Placement Group을 넣지 마십시오([Capacity Blocks — Considerations](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html#capacity-blocks-considerations)).

### 배치 확인과 블록 간 측정

```bash
aws ec2 describe-instance-topology --instance-ids <id> ...
```

결과의 `NetworkNodes` 계층을 아래(인스턴스에 가까운 쪽)부터 비교하면, 어떤 인스턴스들이 같은 spine을 공유하는지 알 수 있습니다. 블록을 여러 개 쓴다면 블록 안 노드 쌍과 블록 사이 노드 쌍으로 각각 `alltoall`·`fi_pingpong`을 측정해 성능 차이를 확인하십시오(아래 "권장 사전 검증 순서").

## InfiniBand → EFA 전환 시 공통 고려사항

IB 환경에서 쓰던 설정이 EFA 환경에 남아 있으면 성능이 크게 떨어지거나 초기화가 실패할 수 있습니다. 전환 시 다음을 점검하십시오.

### NCCL이 EFA를 타는지 반드시 검증

플러그인이 잡히지 않으면 NCCL이 Socket(TCP)으로 떨어져 성능이 크게 낮아집니다. `NCCL_DEBUG=INFO` 로그에서 다음 두 줄을 반드시 확인하십시오.

- `Using network Libfabric`
- `Selected provider is efa ... (found 8 nics)`

### IB용 설정 잔재 제거

최신 NGC 이미지는 OFI 플러그인이 기본값이라 EFA 플러그인이 자동 선택됩니다. 그러나 IB용 설정(`NCCL_NET=IB`, `NCCL_NET_PLUGIN` 변경, `NCCL_IB_*`)이 EFA 잡 스크립트에 남아 있으면 EFA를 쓰지 못할 수 있습니다(`NCCL_NET=IB`를 지정하면 초기화가 실패합니다). EFA 환경에서는 IB용 `NCCL_*` 설정을 모두 제거하십시오. `NCCL_NET`은 지정할 필요가 없으며, 굳이 지정한다면 `Libfabric`입니다.

### 컨테이너의 EFA 구성

최신 NGC 이미지에는 EFA 스택(libfabric, aws-ofi-nccl)이 내장되어 있어 원본 그대로도 정상 성능이 나옵니다. Pyxis/Enroot 환경에서는 `/dev/infiniband`가 별도 마운트 없이 보입니다. 다만 학습 이미지에는 호스트와 버전을 맞추고 DeepEP v2 등 노드 간 백엔드를 대비해 EFA installer 1.50 이상 설치를 권장합니다.

### ConnectX-7 장치 혼동 주의

최신 B200 세대 인스턴스에는 EFA 외에 링크가 연결되지 않은 ConnectX-7 장치 2개가 보일 수 있습니다. NCCL은 EFA를 선택하므로 영향이 없습니다. 이 장치를 보고 IB 경로를 강제하지 마십시오 — 바로 위 "IB용 설정 잔재 제거"의 함정으로 이어집니다.

### GPUDirect RDMA

노드 간 GPU 메모리 직접 전송(GPUDirect RDMA)은 기본 구성에서 활성화되어 있습니다. NCCL 로그에서 `via NET/Libfabric/.../GDRDMA`를 확인하십시오. 이 경로가 꺼져 있으면 노드 간 전송이 호스트 메모리를 경유해 느려집니다.

### rank당 CPU 할당

rank당 CPU가 1코어만 배정되면 EFA가 정상이어도 통신 성능이 크게 떨어질 수 있습니다. rank마다 노드의 vCPU를 GPU 수로 나눈 만큼 충분히 할당해야 합니다. 스케줄러에서 이를 지정하는 방법(예: Slurm의 `--cpus-per-task`)은 [잡 제출](environment/slurm-pyxis-enroot.md) 섹션을 참고하십시오.

!!! note "검증 환경 (2026-10 기준)"
    - CPU 미할당의 영향: `--cpus-per-task` 없이 띄우면 8 GiB all-reduce algbw 92 → 368 GB/s, all-to-all 8.7 → 95 GB/s로 1/4~1/10까지 떨어졌습니다.
    - 권장: `--cpus-per-task=24` (192 vCPU ÷ 8 GPU 기준).
    - 버전 조합(드라이버·CUDA·EFA·aws-ofi-nccl·NCCL)이 어긋나면 성능 저하·오류가 생기므로, 검증된 DLAMI를 쓰고 커널 임의 업그레이드를 피하십시오.

## 대상 하드웨어 사양과 유의점

최신 B200 세대 GPU 인스턴스의 대표 사양은 다음과 같습니다.

!!! note "검증 환경 (2026-10 기준) — P6-B200 사양"
    | 항목 | 값 |
    |---|---|
    | GPU | NVIDIA B200 × 8 (GPU당 메모리 약 180 GB, 인스턴스당 약 1,440 GB) |
    | GPU 간 연결 | NVLink (노드 안 GPU 간 NV18) |
    | EFA | 네트워크 카드 8장, 인스턴스당 3,200 Gbps (GPU당 약 50 GB/s) |
    | vCPU / 메모리 | 192 vCPU / 약 2 TB |
    | 로컬 NVMe | 3.84 TB × 8 (DLAMI는 `/opt/dlami/nvme`에 묶어 자동 마운트, 약 28 TB) |

버전과 무관하게 유효한 유의점은 다음과 같습니다.

- **EFA 대역폭은 직전 세대(H200)와 같은 인스턴스당 3.2 Tbps입니다.** GPU 세대가 바뀌어도 노드 간 통신 대역은 같으므로, 노드를 넘는 all-to-all에 대한 설계 원칙(EP를 노드 안에 두기)은 그대로 유효합니다.
- GPU 메모리 안에서 EP를 노드 수 이하로 나눠 담을 수 있도록 TP·PP와 메모리 예산을 설계하십시오.
- FP8/FP4 사용 시에는 정밀도·수렴과 커널(TransformerEngine 등) 지원 범위를 사전에 확인하십시오. 노드 내 EP 라이브러리 자체는 FP8 dispatch를 지원하지만, Megatron 0.19.0의 `hybridep` 디스패처에서는 FP8 dispatch를 쓸 수 없습니다(Megatron 코드상 제한).

## 권장 사전 검증 순서

본 학습 전에 아래 순서로 네트워크와 통신 경로가 설계대로 동작하는지 확인하고, 대상 구성의 기준선(iteration 시간)을 확보하십시오. **각 단계의 구체적 실행 명령(nccl-tests·`fi_pingpong`·컨테이너 측정)은 [분산 학습 검증](environment/slurm-pyxis-enroot.md#distributed-training-validation)에 정리되어 있습니다.** 여기서는 "무엇을 왜 보는지"만 정리합니다.

1. **호스트 EFA·NCCL 기준선** — 2노드 이상에서 all-reduce·all-to-all 대역폭을 측정해, EFA 경로(`Using network Libfabric`)로 설계 대역폭에 근접하는지 확인.
2. **컨테이너 NCCL 경로** — 학습 이미지 안에서도 호스트와 비슷한 성능이 나오는지 확인. 이미지의 EFA 스택·플러그인 설정이 호스트와 다를 수 있기 때문입니다.
3. **노드 간 지연** — 소형 메시지 왕복 지연(`fi_pingpong`)으로 지연 민감 통신의 기준선을 확보.
4. **EP 디스패처 확인** — MoE를 쓰는 경우 Expert Parallelism 백엔드를 단독으로 확인([MoE / Hybrid-EP 검증](environment/moe-hybrid-ep.md)).
5. **대상 모델·병렬화 구성으로 짧은 학습** — iteration 시간을 기준선으로 기록.
6. (Capacity Block이 여러 개라면) 토폴로지 확인 후 블록 안·블록 사이를 비교.

## 참고 — 공식·1차 문서

**EFA · NCCL**

- [Get started with EFA and NCCL for ML workloads on Amazon EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html) · [EFA 개요](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [Maximize network bandwidth — EFA configuration for P6-B200](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-acc-inst-types.html#efa-for-p6-b200)
- [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) · [libfabric EFA provider (`fi_efa.7`)](https://github.com/ofiwg/libfabric/blob/main/man/fi_efa.7.md) · [`fi_pingpong(1)`](https://ofiwg.github.io/libfabric/main/man/fi_pingpong.1.html)
- [NCCL 문서](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) · [NCCL 환경변수](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) · [nccl-tests](https://github.com/NVIDIA/nccl-tests)

**Capacity Block · 배치**

- [Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html) · [Launch instances using Capacity Blocks](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-launch.html)
- [EC2 instance topology](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-topology.html) · [describe-instance-topology](https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instance-topology.html)

**MoE EP 통신**

- [DeepEP](https://github.com/deepseek-ai/DeepEP) · [DeepEP v2 — AWS 포크](https://github.com/amazon-contributing/DeepEP) · [deepep-v2-benchmark](https://github.com/awslabs/awsome-distributed-ai/tree/main/micro-benchmarks/expert-parallelism/deepep-v2-benchmark)
- [Megatron-LM PR #4793 (DeepEP v2 백엔드, dev)](https://github.com/NVIDIA/Megatron-LM/pull/4793) · [PR #5153 (main 반영)](https://github.com/NVIDIA/Megatron-LM/pull/5153)
- [NVIDIA/nccl #2272 (NCCL 로드맵: EFA GDA support in GIN)](https://github.com/NVIDIA/nccl/issues/2272)

!!! note "출처 유의"
    GitHub·NVIDIA 문서는 AWS 공식 문서가 아닌 1차 소스입니다. 오픈소스의 EFA 지원 상태는 빠르게 바뀌므로 적용 직전에 최신 상태를 확인하십시오.
