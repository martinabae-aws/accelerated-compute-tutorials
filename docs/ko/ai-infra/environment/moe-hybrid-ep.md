---
title: MoE / Hybrid-EP 검증
description: EFA 위 레이어 — 대규모 MoE 학습에서 Expert Parallelism(EP)을 노드 안에 배치하는 설계 원칙과 Hybrid-EP 디스패처 검증 절차
---

# MoE / Hybrid-EP 검증

이 페이지는 EFA 네트워킹 위에 올라가는 **모델·라이브러리 레이어** 를 다룹니다. 네트워크(EFA·NCCL)가 정상임을 확인한 뒤, 대규모 MoE(Mixture-of-Experts) 모델을 학습할 때 **Expert Parallelism(EP)을 어디에 배치하고 어떤 디스패처 백엔드를 쓸지** 를 결정하고 검증하는 단계입니다.

핵심 질문은 하나입니다 — **"Expert 통신(all-to-all)을 EFA(노드 간)로 내보낼 것인가, 노드 안 NVLink에만 묶어 둘 것인가?"** 이 선택이 MoE 학습 성능을 좌우하며, 그래서 EP 설계가 가장 중요한 결정이 됩니다.

!!! note "검증 환경 (2026-10 기준)"
    아래 버전·커밋·명령·기준값은 **P6-B200 2노드(16 GPU) 환경에서 측정한 특정 시점 스냅샷** 입니다. 성능을 보장하는 값이 아니라 참고용 레퍼런스이며, 노드 수·모델·병렬화 구성·소프트웨어 버전에 따라 달라집니다. 원칙(본문)은 버전과 무관하게 유효하므로, 버전이 바뀌면 이 박스만 갱신하면 됩니다.

    - 컨테이너 이미지: `nvcr.io/nvidia/pytorch:26.08-py3` (+ EFA installer 1.50.0)
    - Megatron-LM: 0.19.0
    - Hybrid-EP 라이브러리: 특정 커밋 `94a9f8f6b146c07d97ec58f67cd6d303296d6098` (노드 내 전용 빌드)
    - 검증 병렬화: EP=8, TP=1, PP=1 · 2노드 16 GPU

## 왜 EP를 노드 안에 두는가

MoE는 토큰을 각 expert로 보내고 결과를 돌려받기 위해 **all-to-all** 통신을 씁니다. 모든 rank가 모든 rank와 데이터를 주고받으므로, 데이터 병렬의 all-reduce보다 네트워크 대역폭과 지연에 훨씬 민감합니다.

여기서 결정적인 것은 통신 경로입니다.

- **노드 안(GPU 8개)은 NVLink** — GPU당 한 방향 약 900 GB/s.
- **노드 사이는 EFA** — 인스턴스당 3.2 Tbps(약 400 GB/s), GPU 8개로 나누면 **GPU당 약 50 GB/s**.

즉 노드 내 NVLink와 노드 간 EFA는 한 자릿수 배 이상 대역폭 차이가 납니다. **EP degree가 노드의 GPU 수(8)를 넘으면** all-to-all 트래픽이 NVLink를 벗어나 EFA로 나가고, 이때부터 EP 통신 성능은 EFA 대역폭(GPU당 약 50 GB/s)이 상한이 됩니다.

따라서 설계 원칙은 명확합니다.

!!! tip "설계 원칙"
    1. **EP는 노드 안(≤ 8)에 둔다.** EP 그룹이 한 노드 안에 들어가면 expert 통신이 NVLink에서만 일어나고 EFA를 쓰지 않습니다.
    2. 노드 사이는 **데이터 병렬(DP)·파이프라인 병렬(PP)** 로 확장한다. 노드 간에는 같은 expert를 가진 노드끼리 gradient를 맞추는 DP 통신만 EFA로 오갑니다.
    3. EP 그룹이 한 노드에 들어가려면 **EP × TP ≤ 8** 이어야 하고, 노드당 8 rank를 **연속으로** 배치해야 한다(Megatron은 EP 그룹을 연속된 rank로 묶음).

## 통신 경로 비교 (측정 레퍼런스)

노드 간 all-to-all은 이미 EFA 대역폭의 상한 근처까지 쓰이고 있어, **성능 개선 여지는 튜닝이 아니라 설계(EP를 노드 안에 두는가)에 있습니다.**

!!! note "검증 환경 측정값 (P6-B200 2노드, 16 GPU)"
    | 통신 | 경로 | 측정값 | 의미 |
    |---|---|---|---|
    | all-reduce (16 GiB, nccl-tests) | 노드 안 NVLink + 노드 간 EFA (NCCL NVLS tree) | algbw **374 GB/s** | 노드당 EFA 400 GB/s의 약 94%(계산값) |
    | all-to-all (16 GiB, nccl-tests) | 노드 간 EFA | algbw **95 GB/s** (GPU당 노드 간 트래픽 약 48 GB/s) | GPU당 EFA 몫 50 GB/s의 약 95%(계산값) |
    | Hybrid-EP dispatch / combine (1노드 8 GPU, hidden 7168, 4096 토큰/rank, top-8) | 노드 안 NVLink | API 기준 dispatch 419 GB/s(FP8), combine 512 GB/s | GPU당 노드 간 all-to-all 트래픽보다 수 배~약 10배 수준(측정 방식이 달라 단순 비교는 참고용) |

    - all-reduce(데이터 병렬)는 EFA 대역폭을 거의 다 쓴다.
    - **노드 간 all-to-all도 EFA 상한의 약 95%까지 쓰이고 있다** — 더 끌어올릴 여지가 거의 없다는 뜻.
    - 같은 통신이 **노드 안 NVLink** 에서 일어나면 GPU당 수백 GB/s로, 노드 간보다 한 자릿수 배 빠르다.

## 권장 구성: EP=8 + Hybrid-EP

가장 단순하면서 성능이 좋은 구성은 **노드당 GPU 8개 = 하나의 EP 그룹(EP=8, TP=1)** 입니다. 이 구성에서는 expert 통신(all-to-all)이 노드 안 NVLink에서만 일어나고 EFA를 전혀 쓰지 않습니다. 노드 사이에는 같은 expert를 가진 노드끼리 gradient를 동기화하는 데이터 병렬 통신만 EFA로 오갑니다.

```
node1: GPU 0~7 = EP 그룹 (expert를 8개 GPU가 나눠 가짐)
node2: GPU 0~7 = EP 그룹 (node1과 같은 expert 구성)
노드 사이: 같은 expert끼리 gradient 동기화 (데이터 병렬, EFA)
```

### Hybrid-EP는 노드 내 전용으로 빌드한다

EP=8에서는 expert 통신이 노드를 넘지 않으므로, Hybrid-EP 디스패처를 **노드 내 전용(멀티노드 백엔드 없음)으로 빌드하면 됩니다.** 노드 간 백엔드(NIXL·DOCA 등)는 쓰이지 않습니다.

```bash
export TORCH_CUDA_ARCH_LIST=10.0
git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP
git checkout <특정 커밋>
DISABLE_AGGRESSIVE_PTX_INSTRS=1 pip install --no-build-isolation --no-deps .
```

- `--no-deps`: 컨테이너 이미지에 이미 있는 PyTorch 등 의존 패키지를 pip가 다시 설치하지 않게 합니다.
- `HYBRID_EP_MULTINODE` 를 설정하지 않으면 노드 내 전용으로 빌드됩니다. 빌드 로그에 `Multinode enabled`, `NIXL path`, `DOCA path` 가 나오지 **않으면** 노드 내 전용으로 올바르게 빌드된 것입니다.
- **B200 계열에서는 `TORCH_CUDA_ARCH_LIST=10.0` 을 직접 지정하세요.** 지정하지 않으면 다른 아키텍처 기본값으로 빌드될 수 있습니다.

!!! note "검증 환경 (2026-10 기준)"
    - 커밋: `94a9f8f6b146c07d97ec58f67cd6d303296d6098`
    - B200 기준 빌드 시간 약 1분 30초.
    - 노드 내 단독 확인(1노드 8 GPU): `python tests/test_hybrid_ep.py --num-processes 8` → `PASS` 3개와 대역폭이 출력됩니다.

### Megatron 디스패처 설정

사용자 학습 코드(Megatron 계열)에서 flex 디스패처의 Hybrid-EP 백엔드를 지정합니다.

```
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep --expert-model-parallel-size 8
```

- EP 그룹을 한 노드에 묶으려면 Slurm에서 노드당 8 rank를 연속 배치해야 합니다(`--ntasks-per-node=8`). 검증은 TP=1, EP=8로 수행했습니다.
- `--moe-grouped-gemm` 을 쓸 때는 `--disable-bias-linear` 가 필요합니다. 없으면 `AssertionError: bias_dropout_fusion is not supported in TEGroupedMLP when add_bias_linear=True` 로 모델 생성이 실패합니다.

!!! warning "FP8 디스패치 제약"
    Hybrid-EP 라이브러리 자체는 FP8 dispatch를 지원하지만, **Megatron 0.19.0의 `hybridep` 디스패처에서는 FP8 dispatch를 쓸 수 없습니다(Megatron 코드상 제한). BF16을 사용하세요.** FP8/FP4로 학습할 계획이라면 디스패처·커널(TransformerEngine 등)의 정밀도 지원 범위를 사전에 확인해야 합니다. (버전 스냅샷 — 상위 버전에서 바뀔 수 있음)

### 기대 결과

학습이 정상 진행되어야 합니다 — loss가 정상적으로 감소하고 NaN이 없어야 합니다. NCCL 로그에서 DP 통신이 EFA(`Using network Libfabric`)로 잡히고, 노드 내 전용 빌드이므로 NIXL·DOCA 등 노드 간 백엔드는 쓰이지 않습니다.

!!! note "검증 환경 측정값 (2노드, EP=8, DP=16)"
    작은 시험 모델(4층, expert 16개)을 30 iteration 돌린 결과입니다. **실제 모델의 디스패처 간 성능 차이를 나타내지는 않습니다.**

    - `hybridep` 디스패처: 30 iteration 정상, 마지막 10회 평균 **167.8 ms/iteration**
    - 같은 설정의 `alltoall` 디스패처: 184.9 ms/iteration (약 9% 느림)

## 검증용 실행 예시

노드 내 Hybrid-EP가 EP=8에서 정상 동작하는지 확인하는 작은 시험 모델 설정입니다(실제 학습 설정이 아닙니다). 런처가 Slurm 변수로 `RANK`·`WORLD_SIZE`·`LOCAL_RANK` 를 채웁니다.

```bash
#!/bin/bash
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
export MASTER_ADDR=${MASTER_ADDR:?} MASTER_PORT=${MASTER_PORT:-29600}
export CUDA_DEVICE_MAX_CONNECTIONS=1
cd <사용자 학습 코드 경로>
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

본 학습 전에는 **대상 모델·병렬화 구성 그대로** 짧게(수십 iteration) 돌려 iteration 시간을 기준선으로 확보해 두는 것을 권장합니다. 노드 수를 늘려 측정할 때 값이 크게 떨어지면 이 기준선과 비교해 문제를 조기에 발견할 수 있습니다.

## EP > 8로 확장할 경우 (참고)

권장 구성(EP=8)에는 해당하지 않지만, 향후 EP를 노드 밖으로 확장할 때를 위한 참고 내용입니다.

### 노드 내 전용 빌드에서 EP가 노드를 넘으면

노드 내 전용으로 빌드한 Hybrid-EP에서 EP 그룹이 한 노드를 넘으면(EP > 8, 또는 EP × TP > 8), 성능이 조용히 떨어지는 것이 아니라 **학습 시작 시점에 아래 메시지와 함께 바로 종료됩니다.**

```
Inter-node communication is not supported. Please rebuild with HYBRID_EP_MULTINODE flag, group_size=16, buffer_config.num_of_ranks_per_node=8.
```

조치: EP ≤ 8로 조정하거나, 아래 멀티노드 경로 중 하나를 선택합니다.

### 노드 간 EP가 꼭 필요하면

- **AWS가 EFA용으로 지원하는 경로는 DeepEP v2(AWS 포크, [amazon-contributing/DeepEP](https://github.com/amazon-contributing/DeepEP))입니다.** 사용자 학습 코드(Megatron 계열)에서는 `--moe-flex-dispatcher-backend deepepv2` 를 씁니다. 이 백엔드는 현재 Megatron `dev` 브랜치에만 있어, 안정 버전 기반 포크에는 반영 작업이 필요할 수 있습니다. 컨테이너에는 NCCL 2.31 이상과 EFA installer 1.50 이상, 호스트에는 EFA 커널 드라이버 3.3.0 이상이 필요합니다(권장 DLAMI는 호스트 조건을 충족).
- 안정 버전(0.19.0)의 `deepep` 백엔드(원본 DeepEP)는 EFA에서 노드 간으로 동작하지 않습니다.
- Hybrid-EP를 멀티노드로 빌드해 노드 간에 쓰는 경우, EFA에서는 GPU 직접 경로가 아니라 CPU를 거치는 방식이라 성능이 낮습니다.
- 어떤 라이브러리든 **노드 간 all-to-all은 EFA 대역폭(GPU당 약 50 GB/s)이 상한** 입니다. 그래서 가능하면 EP를 노드 안에 두고, 확장은 DP·PP로 하는 것이 유리합니다.

!!! warning "오픈소스 지원 범위"
    Hybrid-EP(DeepEP)는 NVIDIA/DeepSeek 오픈소스이며 **AWS 지원 대상이 아닙니다.** DeepEP v2(AWS 포크) 역시 오픈소스이고, 오픈소스의 EFA 지원 상태는 빠르게 바뀌므로 적용 직전에 최신 상태를 확인하세요.

## 하드웨어 메모리 설계 참고

B200 계열은 GPU당 메모리가 약 180 GB(인스턴스당 약 1,440 GB)입니다. GPU 메모리 안에서 EP=8로 expert를 나눠 담을 수 있도록 TP·PP와 메모리 예산을 함께 설계하세요. 노드 간 EFA 대역폭은 GPU 세대가 바뀌어도 인스턴스당 3.2 Tbps로 같으므로, 노드를 넘는 all-to-all에 대한 설계 원칙(EP를 노드 안에 두기)은 그대로 유지됩니다.

## 요약

1. **EP는 노드 안(≤ 8)에 두고**, 노드 사이는 데이터 병렬·파이프라인 병렬로 확장한다. (EP × TP ≤ 8, 노드당 8 rank 연속 배치)
2. EP=8에서는 **Hybrid-EP를 노드 내 전용으로 빌드** 하고, Megatron 디스패처는 `hybridep` 백엔드를 쓴다. Megatron 0.19.0에서는 FP8 dispatch 불가 — BF16 사용.
3. 노드 내 단독 테스트(`test_hybrid_ep.py`)로 디스패처를 먼저 확인하고, 본 학습 전에 대상 구성 그대로 짧게 돌려 iteration 시간을 기준선으로 확보한다.
4. EP > 8이 필요하면 DeepEP v2(AWS 포크)를 검토한다. Hybrid-EP·DeepEP는 오픈소스로 AWS 지원 대상이 아니다.
