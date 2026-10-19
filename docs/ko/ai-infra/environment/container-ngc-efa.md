---
title: 컨테이너 이미지 (NGC + EFA)
description: NGC PyTorch 베이스 이미지에 AWS EFA 스택을 더해 대규모 GPU 학습용 컨테이너 이미지를 구성하는 방법 — 이미지 빌드 함정과 EFA 통신 스택 중심.
---

# 컨테이너 이미지 (NGC + EFA)

대규모 GPU 학습 클러스터에서는 모든 노드가 **동일한 소프트웨어 환경** 을 공유해야 재현성과 성능을 보장할 수 있습니다. 이 환경을 호스트 OS에 직접 설치하는 대신, NVIDIA NGC가 배포하는 PyTorch 컨테이너 이미지를 베이스로 삼고 그 위에 AWS EFA(Elastic Fabric Adapter) 통신 스택을 더해 학습 이미지를 만듭니다. 이렇게 만든 이미지를 `.sqsh` 아카이브로 공유 스토리지에 두면, 각 노드에서 이를 꺼내 학습 잡을 실행합니다(스케줄러로 잡을 띄우는 방법은 사용하는 오케스트레이터 가이드를 따릅니다).

이 페이지는 다음을 다룹니다.

- NGC PyTorch 이미지에 이미 들어 있는 EFA 스택과 그 한계
- NGC 베이스 위에 EFA installer를 더하는 Dockerfile 패턴과 함정
- 컨테이너용 EFA installer 플래그, `LD_LIBRARY_PATH`, MPI 충돌 회피
- 이미지를 Enroot `.sqsh`로 가져와 노드에서 재사용하는 방법

!!! note "검증 환경 (2026-10 기준)"
    아래 버전·수치는 특정 시점에 P6-B200 2노드로 검증한 **참고 스냅샷** 입니다. 원칙은 버전과 무관하게 유효하며, 버전이 바뀌면 이 박스만 갱신하면 됩니다.

    - 베이스 이미지: `nvcr.io/nvidia/pytorch:26.08-py3` (NCCL 2.30.7 · CUDA 13.4.1 · PyTorch 2.14)
    - 추가한 EFA installer: 1.50.0 (libfabric 2.6.0, aws-ofi-nccl 1.21.1)
    - 스케줄러: Slurm 23.11.4 · Pyxis 0.24.0 · Enroot 4.2.1
    - 호스트 AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04), 드라이버 595.91.07 / CUDA 13.2
    - 검증 이미지 크기: 약 25 GB (`.sqsh`)

## 왜 NGC 베이스인가

NGC PyTorch 이미지는 CUDA, cuDNN, NCCL, PyTorch가 서로 호환되도록 함께 빌드되어 배포됩니다. 버전 조합을 직접 맞추는 수고를 덜 수 있고, NVIDIA가 B 시리즈 GPU에 맞춘 최적화를 미리 반영해 둡니다.

호스트 드라이버보다 컨테이너의 CUDA 버전이 더 높아도, **CUDA Forward Compatibility 모드** 로 동작합니다. 컨테이너 시작 시 `CUDA Forward Compatibility mode ENABLED`가 표시되며, GPU 연산·NCCL/EFA 통신·학습까지 정상 동작합니다. 다만 사용자 학습 코드에 커스텀 CUDA 커널이 포함된 경우, 해당 환경에서 별도로 동작을 확인하는 것이 안전합니다.

## NGC 이미지에 이미 들어 있는 EFA 스택

최근 NGC PyTorch 이미지에는 AWS EFA 스택이 이미 포함되어 있습니다. 검증한 이미지에서 확인한 구성은 다음과 같습니다.

| 항목 | 내용 |
|---|---|
| libfabric | `/opt/amazon/efa` 에 포함 (검증 시 2.4.0) |
| aws-ofi-nccl | `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so` (검증 시 1.19.0) |
| 기본 환경변수 | `NCCL_NET_PLUGIN=ofi`, `NCCL_TUNER_PLUGIN=ofi` |
| 라이브러리 경로 | `/etc/ld.so.conf.d/` 에 등록되어 `LD_LIBRARY_PATH`를 따로 잡지 않아도 찾음 |

핵심 원칙은 다음과 같습니다.

- **원본 이미지 그대로도 EFA로 정상 성능이 나옵니다.** 호스트의 EFA 라이브러리나 `/dev/infiniband`를 `--container-mounts`로 마운트할 필요가 없습니다. 이미지 안의 EFA 스택이 호스트 EFA 장치를 그대로 사용합니다.
- 호스트 라이브러리를 컨테이너에 마운트하고 `LD_LIBRARY_PATH`에 끼워 넣는 방식은 **권장하지 않습니다.** 이미지 안 EFA 스택과 호스트 라이브러리의 버전이 섞이면 어긋날 수 있습니다. 버전을 맞추려면 호스트 마운트가 아니라, 뒤에서 설명하는 이미지 빌드 방식을 사용하십시오.
- **`NCCL_NET_PLUGIN=ofi`를 지우거나 바꾸지 마십시오.** 기본 이름의 `libnccl-net.so`는 이미지 안의 다른(InfiniBand용) 플러그인을 가리킵니다. `ofi`를 지정해야 EFA 경로가 선택됩니다.

!!! warning "이미지 안의 EFA 버전이 호스트와 다를 수 있음"
    원본 NGC 이미지의 EFA 버전은 호스트 AMI에 설치된 EFA installer 버전과 다를 수 있습니다. 성능은 문제없더라도, 버전을 맞춰 두면 트러블슈팅이 단순해지고, EP(Expert Parallelism) 확장용 통신 라이브러리처럼 특정 EFA installer 버전 이상을 전제로 하는 구성요소를 쓸 수 있게 됩니다. 그래서 아래 빌드 방식을 권장합니다.

## 권장: 학습 이미지에 EFA installer를 더한다

학습 이미지에는 호스트 AMI와 **같은 버전의 EFA installer** 를 설치하기를 권장합니다. 성능은 원본과 같지만, 호스트와 버전이 맞고(libfabric, aws-ofi-nccl), EP 확장용 통신 라이브러리의 전제 조건(특정 EFA installer 버전 이상)을 충족합니다.

```dockerfile
FROM nvcr.io/nvidia/pytorch:26.08-py3
ARG EFA_INSTALLER_VERSION=1.50.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates git \
 && cd /tmp \
 && curl -fsSL -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
 && tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
 && cd aws-efa-installer \
 && ./efa_installer.sh -y -g --skip-kmod --skip-limit-conf --no-verify \
 && cd / && rm -rf /tmp/aws-efa-installer* /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}
```

### 함정 ① — 같은 RUN 안에서 `apt-get update`를 먼저

**EFA installer를 실행하기 전에, 같은 `RUN` 레이어 안에서 `apt-get update`를 먼저 수행해야 합니다.** EFA installer는 실행 도중 apt로 서드파티 의존 패키지를 설치합니다. apt 패키지 목록이 비어 있으면 다음과 같이 실패합니다.

```
E: Unable to locate package environment-modules
...
failed to install third-party dependencies
```

Docker 레이어는 캐시되므로, 별도 `RUN`에서 `apt-get update`를 돌려도 EFA installer가 실행되는 레이어에서는 목록이 비어 있을 수 있습니다. 그래서 **`apt-get update`와 EFA installer 실행을 하나의 `RUN`으로 묶는 것** 이 안전합니다.

### 함정 ② — 컨테이너용 EFA installer 플래그

컨테이너 안에는 커널도, 실제 EFA 장치도 없습니다. 호스트에서와 같은 설치 과정을 그대로 돌리면 실패하므로, 컨테이너용 플래그를 지정합니다.

| 플래그 | 의미 | 이유 |
|---|---|---|
| `--skip-kmod` | 커널 모듈 설치 건너뜀 | 커널 모듈은 호스트가 담당. 컨테이너에서는 설치 불가·불필요 |
| `--skip-limit-conf` | `limits.conf`(ulimit) 수정 건너뜀 | memlock 등 ulimit은 호스트에서 설정 |
| `--no-verify` | 설치 후 장치 검증 건너뜀 | 빌드 환경에는 EFA 장치가 없음 |
| `-g` | GPUDirect 관련 구성 포함 | GPU 간/노드 간 RDMA 경로 |

- EFA installer는 NGC 베이스를 감지해 **NGC용 aws-ofi-nccl 패키지**(예: `libnccl-ofi-ngc-v3`)를 설치합니다. 따라서 `--disable-ngc`는 필요하지 않습니다.

### 함정 ③ — `LD_LIBRARY_PATH`

EFA installer가 설치한 라이브러리를 런타임에서 확실히 찾도록, `/opt/amazon/ofi-nccl/lib`과 `/opt/amazon/efa/lib`을 `LD_LIBRARY_PATH` 앞쪽에 둡니다.

```dockerfile
ENV LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}
```

이렇게 하면 이미지 안의 EFA/aws-ofi-nccl이 우선 사용되어, 베이스 이미지에 원래 있던 구 버전 라이브러리와 섞이는 것을 피할 수 있습니다.

### 함정 ④ — NGC Open MPI 위에 다른 MPI를 얹지 말 것

NGC 이미지에는 이미 Open MPI 5(`/usr/local/mpi`)가 들어 있습니다. 여기에 EFA installer가 가져오는 **다른 Open MPI로 빌드한 프로그램** 을 컨테이너 안에서 실행하면 두 Open MPI가 섞여 다음과 같이 실패합니다.

```
opal_shmem_base_select failed
--> Returned value -1 instead of OPAL_SUCCESS
...
MPI_INIT failed
```

**원칙: NGC 이미지의 MPI를 그대로 두고, 그 위에 다른 Open MPI를 얹지 마십시오.** 컨테이너 안에서 노드 간 통신을 검증할 때는 MPI에 의존하는 벤치마크 대신 **MPI가 필요 없는 PyTorch 기반 측정** 을 사용하는 것을 권장합니다. (호스트에서의 `nccl-tests`는 호스트 Open MPI와 `srun --mpi=pmix`를 쓰므로 이 문제와 무관합니다.)

!!! note "검증 환경 (2026-10 기준) — 전체 참고 Dockerfile"
    아래는 검증 구성을 정리한 전체 참고 예시입니다. 1단계(EFA)에 더해, 노드 내 전용으로 빌드한 MoE 통신 라이브러리(2단계)와 사용자 학습 코드(3단계)를 얹는 형태입니다. 실 환경의 학습 이미지에 맞게 필요한 단계만 가져다 쓰십시오. 커밋 해시 등 특정 시점 값은 이 박스에만 유지합니다.

    ```dockerfile
    FROM nvcr.io/nvidia/pytorch:26.08-py3

    ARG EFA_INSTALLER_VERSION=1.50.0
    ARG HYBRID_EP_COMMIT=94a9f8f6b146c07d97ec58f67cd6d303296d6098

    # 1. EFA installer (컨테이너용: 커널 모듈 제외). 같은 RUN에서 apt-get update 먼저.
    RUN apt-get update \
     && apt-get install -y --no-install-recommends curl ca-certificates git \
     && cd /tmp \
     && curl -fsSL -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
     && tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
     && cd aws-efa-installer \
     && ./efa_installer.sh -y -g --skip-kmod --skip-limit-conf --no-verify \
     && cd / && rm -rf /tmp/aws-efa-installer* /var/lib/apt/lists/*

    ENV LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}

    # 2. MoE 통신 라이브러리 (노드 내 전용 빌드, B200)
    ENV TORCH_CUDA_ARCH_LIST=10.0
    RUN git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP \
     && cd /opt/DeepEP && git checkout ${HYBRID_EP_COMMIT} \
     && DISABLE_AGGRESSIVE_PTX_INSTRS=1 pip install --no-build-isolation --no-deps . \
     && rm -rf /opt/DeepEP/build

    # 3. 사용자 학습 코드(학습 프레임워크 포크, 커스텀 커널 등)를 이 아래에 추가
    ```

## 이미지 빌드와 Enroot 가져오기

이미지를 빌드한 뒤, Enroot로 `.sqsh` 아카이브로 변환해 공유 스토리지에 둡니다.

```bash
sudo docker build -t <image>:<tag> .
sudo enroot import -o /<shared-path>/<image>.sqsh dockerd://<image>:<tag>
```

### 왜 .sqsh를 공유 스토리지에 두는가

- `.sqsh` 파일을 공유 스토리지에 두고 `--container-image=/<shared-path>/<image>.sqsh`로 지정하면, **각 노드가 레지스트리(NGC 등)에서 직접 받지 않아도 됩니다.** 수십 노드가 동시에 레지스트리에서 이미지를 받으면 생기는 부하와 실패 위험을 피할 수 있습니다.
- Enroot는 잡을 시작할 때마다 `.sqsh`를 노드의 로컬 NVMe에 풉니다. 한 할당(`salloc`/`sbatch`) 안에서 여러 `srun`을 실행할 때는 `--container-name=<이름>`으로 컨테이너를 재사용하면, 두 번째 `srun`부터는 다시 풀지 않고 바로 시작됩니다.

```bash
salloc -N <노드 수> --ntasks-per-node=8 --cpus-per-task=24 --gres=gpu:8
# 첫 srun: .sqsh를 로컬 NVMe에 풀어 컨테이너 생성
srun --container-image=/<shared-path>/<image>.sqsh --container-name=train \
     --container-mounts=/<shared-path>:/<shared-path> bash -c 'nvidia-smi -L | wc -l'
# 두 번째 srun부터: 이름으로 재사용 → 바로 시작
srun --container-name=train --container-mounts=/<shared-path>:/<shared-path> python /<shared-path>/<path>/train.py
```

### Enroot가 컨테이너에서 EFA 장치를 보게 하기

호스트 Enroot 설정에서 `ENROOT_RESTRICT_DEV n`이면, 컨테이너 안에서 `/dev/infiniband`(EFA)가 별도 마운트 없이 그대로 보입니다. 따라서 EFA 통신을 위해 `--container-mounts`로 장치를 수동 마운트할 필요가 없습니다. (Enroot/Pyxis 호스트 측 설정은 스케줄러 구성 페이지를 참조하십시오.)

## 동작 확인

이미지가 노드에서 제대로 뜨고 GPU·EFA 장치가 보이는지 간단히 확인합니다. 아래는 Slurm + Pyxis 예시이며, 다른 오케스트레이터에서는 해당 방식으로 컨테이너를 띄우면 됩니다.

```bash
srun -N2 --ntasks-per-node=1 --gres=gpu:8 \
  --container-image=/<shared-path>/<image>.sqsh \
  bash -c 'hostname; nvidia-smi -L | wc -l; ls /dev/infiniband | head -3'
```

통신 성능 측정은 별도 검증 단계에서 수행합니다 — 컨테이너 안에서 **PyTorch 기반 측정**(MPI 불필요)으로 하며, 자세한 절차는 [분산 학습 검증](slurm-pyxis-enroot.md#distributed-training-validation)을 참고하십시오. NCCL 로그(`NCCL_DEBUG=INFO`)에서 다음이 보이면 EFA 경로입니다.

```
NET/OFI Selected provider is efa, fabric is efa-direct (found 8 nics)
Using network Libfabric
```

- `Using network Libfabric` + `provider is efa`가 보여야 EFA 경로입니다.
- `Using network Socket`이 보이면 TCP로 폴백한 것으로, 성능이 크게 떨어집니다. `NCCL_NET_PLUGIN=ofi`가 유지되는지, InfiniBand용 `NCCL_*` 설정이 남아 있지 않은지 확인하십시오.
- 인스턴스의 EFA 장치 수만큼 `found N nics`가 보여야 합니다(예: EFA 8개인 인스턴스는 `found 8 nics`).

!!! tip "성능이 기준값보다 크게 낮다면"
    EFA 경로(`Using network Libfabric`)인데도 통신 대역폭이 기준값의 1/4~1/10이라면, rank당 CPU 코어가 부족한 경우가 많습니다. `--cpus-per-task=24`처럼 rank마다 충분한 코어를 할당하십시오. 검증 환경에서는 컨테이너 2노드 측정 시 `--cpus-per-task=24`가 필수였습니다.

## 자주 겪는 문제 요약

| 증상 | 원인 | 조치 |
|---|---|---|
| `docker build`에서 `E: Unable to locate package environment-modules` | EFA installer 실행 전에 apt 목록이 비어 있음 | 같은 `RUN`에서 `apt-get update`를 먼저 |
| 컨테이너 안 MPI 프로그램이 `opal_shmem_base_select failed` → `MPI_INIT failed` | NGC Open MPI와 다른 Open MPI가 섞임 | NGC 이미지에 다른 Open MPI를 얹지 않음. 통신 확인은 PyTorch 기반 측정 사용 |
| NCCL 로그에 `Using network Socket` | EFA 플러그인 미로드(TCP 폴백) | `NCCL_NET_PLUGIN=ofi` 유지 확인, IB용 `NCCL_*` 설정 제거 |
| EFA는 타는데(`Libfabric`) 대역폭이 기준값의 1/4~1/10 | rank당 CPU 코어 부족 | `--cpus-per-task` 상향(예: 24) |
| 이미지 안 EFA 버전이 호스트와 다름 | 원본 NGC 내장 EFA 사용 | 학습 이미지에 호스트와 같은 EFA installer를 설치해 버전 일치 |
| 컨테이너 시작 시 `CUDA Forward Compatibility mode ENABLED` | 컨테이너 CUDA가 호스트 드라이버 CUDA보다 높음 | 정상 동작. 커스텀 커널은 해당 환경에서 별도 확인 |
