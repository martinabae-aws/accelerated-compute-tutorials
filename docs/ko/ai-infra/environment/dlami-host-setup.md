---
title: DLAMI 호스트 구성·검증
description: 대규모 GPU 학습 클러스터의 호스트 레이어를 DLAMI 기반으로 구성하는 방법 — DLAMI 계열 선택 기준과, GPU·드라이버·EFA·GDRCopy 등 사전 설치 구성요소 검증 (오케스트레이터·스토리지 무관 공통 절차)
---

# DLAMI 호스트 구성·검증

이 페이지는 GPU 학습 클러스터의 **호스트 레이어** 를 다룹니다. 각 GPU 노드가 공통으로 갖추어야 하는 운영체제·NVIDIA 드라이버·네트워크(EFA) 스택을 어떤 AMI로 확보하고, 기동 후 그 구성이 올바른지 어떻게 검증하는지를 설명합니다. 스케줄러·컨테이너 런타임·학습 런타임은 이 호스트 레이어 위에 얹히므로(오케스트레이터가 Slurm이든 EKS든 ParallelCluster든 공통), 여기서 호스트가 깨끗하게 검증되어야 상위 레이어의 문제를 빠르게 분리할 수 있습니다.

핵심 메시지는 하나입니다. **GPU·EFA·고속 스토리지에 필요한 저수준 구성요소를 손으로 쌓지 말고, 이미 서로 버전이 맞춰진 상태로 묶여 배포되는 AMI를 그대로 쓰라는 것** 입니다. 그리고 그 정합성을 깨는 가장 흔한 실수가 커널 업그레이드이므로, 이 페이지는 "커널을 올리지 않는다"는 원칙을 반복해서 강조합니다.

## DLAMI 종류와 선택

AWS Deep Learning AMI(DLAMI)는 하나가 아니라 **여러 계열** 이 있습니다. 어떤 계열을 고르느냐에 따라 사전 설치되는 소프트웨어와 운영 방식이 달라지므로, 호스트 구성을 시작하기 전에 워크로드에 맞는 계열을 먼저 결정해야 합니다.

### 계열 구분

DLAMI는 크게 **무엇이 미리 들어있는가** 로 나뉩니다.

- **Base 계열** — NVIDIA 드라이버·CUDA·EFA·컨테이너 툴킷 등 **하부 스택만** 포함하고, 학습 프레임워크(PyTorch 등)는 포함하지 않습니다. 프레임워크는 컨테이너로 올리는 것을 전제로 합니다.
- **Framework 계열** — 특정 프레임워크(PyTorch 등)가 드라이버·CUDA와 함께 **미리 설치** 되어 있습니다. 컨테이너 없이 호스트에서 바로 학습을 돌리는 구성에 적합합니다.
- **Multi-framework / 기타 계열** — 여러 프레임워크를 번들하거나, 최소 구성만 담는 등 용도별 변형이 있습니다.

> DLAMI 계열·이름·구성은 AWS가 주기적으로 갱신하므로, 실제 선택 시 [DLAMI Developer Guide](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)에서 최신 목록을 확인하십시오.

### 의사결정 포인트

대규모 GPU 학습 **호스트 레이어** 에 어떤 DLAMI가 맞는지는 다음을 기준으로 판단합니다.

- **학습 프레임워크를 컨테이너로 올리는가, 호스트에 직접 설치하는가** — 컨테이너로 올린다면 프레임워크가 없는 Base 계열이 가볍고 재현성이 높습니다. 호스트에서 바로 돌린다면 Framework 계열이 편합니다.
- **노드 간 모든 노드의 환경을 동일하게 유지해야 하는가** — 대규모 분산 학습은 모든 노드가 같은 소프트웨어 환경이어야 합니다. 환경을 컨테이너 이미지로 고정하면 호스트는 가볍게 유지할 수 있습니다.
- **EFA·GDRCopy·고속 스토리지 커널 모듈이 드라이버·커널과 정합되어 있는가** — 어떤 계열을 쓰든, 아래 "호스트에 필요한 요건"을 충족하는지가 핵심입니다.
- **운영용 AMI를 직접 구워 재사용하는가** — 베이스 위에 스케줄러·런타임을 얹어 운영용 AMI로 굽는다면, 커널·커널 모듈이 고정(hold)된 계열이 유지보수에 유리합니다.

### 호스트에 필요한 요건 (계열 무관 공통)

어떤 DLAMI(또는 커스텀 AMI)를 쓰든, GPU 학습 호스트에는 서로 **버전 의존성이 강한** 구성요소가 **실행 중인 커널과 정합된 상태로** 갖춰져야 합니다.

- **NVIDIA 드라이버** 와 그 위의 CUDA 런타임
- 멀티‑GPU를 하나의 패브릭으로 묶는 **Fabric Manager** (NVLink/NVSwitch 토폴로지 관리)
- 노드 간 고속 통신을 담당하는 **EFA 커널 드라이버 · libfabric · aws-ofi-nccl**
- GPU 메모리 직접 전송을 가능케 하는 **GDRCopy**
- 컨테이너에서 GPU를 노출하는 **NVIDIA Container Toolkit**
- (FSx for Lustre를 쓰는 경우) 공유 스토리지 마운트를 위한 **Lustre client 커널 모듈**

특히 **EFA 커널 드라이버**(그리고 FSx for Lustre를 쓴다면 Lustre client)는 커널 모듈이므로, 커널이 바뀌면 해당 모듈이 더 이상 맞지 않아 네트워크 통신이나 스토리지 마운트가 깨질 수 있습니다. 이 모든 것을 손으로 설치해 버전을 맞추는 것은 재현성이 떨어지고 실수가 쉽습니다. 그래서 **이 요건을 이미 충족한 상태로 묶여 배포되는 AMI를 그대로 쓰는 것** 이 안전합니다.

!!! info "이 가이드가 다루는 범위"
    아래부터는 **Base OSS 계열(Deep Learning Base OSS Nvidia Driver GPU AMI)을 선택한 경우** 의 호스트 구성·검증을 다룹니다. 이 계열은 위 요건을 이미 충족하며, 학습 프레임워크를 컨테이너로 올리는 대규모 분산 학습 구성에 적합합니다. 다른 DLAMI 계열에 대한 가이드는 추후 추가될 예정입니다.

## Base OSS 계열 — 구성과 특징

AWS가 제공하는 **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)** 는 위 요건을 **이미 서로 맞춰진 상태로 묶어** 제공합니다. "Base" 계열이므로 특정 프레임워크(PyTorch 등)는 포함하지 않고 드라이버·EFA·툴킷 등 하부 스택만 담겨 있어, 학습 프레임워크는 컨테이너로 올리는 구성에 적합합니다. 이 AMI를 기준으로 삼고, 그 위에 스케줄러·컨테이너 런타임 등 상위 레이어를 추가해 운영용 AMI를 만드는 것이 권장 흐름입니다.

!!! note "검증 환경 (2026-10 기준)"
    AMI ID는 리전별로 다르므로 SSM Public Parameter로 조회합니다.

    ```bash
    aws ssm get-parameter --region <region> \
      --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
      --query Parameter.Value --output text
    ```

    검증에 사용한 빌드는 `20261002` 이며, 다음 구성이 사전 설치되어 있었습니다.

    | 구성요소 | 버전 |
    |---|---|
    | 커널 | 7.0.0-1014-aws |
    | NVIDIA 드라이버 / CUDA(드라이버 기준) | 595.91.07 / 13.2 |
    | Fabric Manager | active |
    | EFA 드라이버 / EFA installer | 3.3.0 / 1.50.0 (libfabric 2.6.0, aws-ofi-nccl 1.21.1, Open MPI 4.1.7·5.0.9, PMIx 4.2.8) |
    | GDRCopy | `gdrdrv` 로드, `/dev/gdrdrv` |
    | NVIDIA Container Toolkit / Docker | 1.20.1 / 29.8.2 |
    | Lustre client | 2.15.6 (현재 커널용 모듈) |
    | nccl-tests | `/usr/local/cuda-XX/efa/test-cuda-XX/` (CUDA 12.8·12.9·13.0·13.2) |
    | 로컬 NVMe | 8개를 LVM으로 묶어 `/opt/dlami/nvme`(약 28 TB)에 자동 마운트 |

    위 버전은 특정 시점의 스냅샷입니다. AMI 빌드가 갱신되면 각 구성요소 버전도 바뀌므로, 실제 환경에서 조회한 값을 기준으로 삼으십시오.

> 📎 [DLAMI Developer Guide](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)

## 커널을 업그레이드하지 않는다 (가장 중요한 원칙)

DLAMI에서는 **커널과 커널 모듈 패키지가 고정(hold)** 되어 있습니다. 이는 의도된 설계입니다.

- EFA 커널 드라이버는 **실행 중인 커널 버전에 묶인 커널 모듈** 입니다. (FSx for Lustre를 쓰는 경우 Lustre client도 마찬가지입니다.)
- 커널을 올리면 새 커널에 맞는 모듈이 없어, **EFA 통신이 깨지거나**(공유 스토리지로 FSx for Lustre를 쓴다면 그 마운트도) 실패할 수 있습니다.
- 따라서 `apt upgrade` 등으로 커널을 올리지 말고, hold 상태를 유지해야 합니다.

이 원칙은 운영용 AMI를 새로 구울 때도 그대로 적용됩니다. **운영용 AMI를 만들 때도 커널 버전을 고정하십시오.** 보안 패치가 필요하면 커널을 올리는 대신, 새 DLAMI 빌드로 베이스를 교체하는 방식(해당 빌드에서 모든 모듈이 다시 정합화된 상태)을 택하는 것이 안전합니다.

!!! warning "커널 변경이 유발하는 전형적 증상"
    - EFA 디바이스 미인식 또는 통신 초기화 실패 — EFA 커널 드라이버 불일치
    - (FSx for Lustre 사용 시) 마운트 실패 — Lustre 커널 모듈이 새 커널과 맞지 않음

    문제가 발생하면 가장 먼저 `uname -r` 로 실행 커널이 AMI 기본 커널에서 바뀌지 않았는지 확인하십시오.

## CUDA Forward Compatibility — 컨테이너 CUDA가 호스트보다 높아도 되는 이유

학습 프레임워크는 컨테이너로 올리므로, **컨테이너 안의 CUDA 버전이 호스트 드라이버의 CUDA 버전보다 높을 수 있습니다.** 이때 작동하는 것이 **CUDA Forward Compatibility** 입니다.

- NVIDIA 드라이버는 자신보다 더 새로운 CUDA 런타임을 Forward Compatibility 패키지를 통해 지원합니다.
- 즉 호스트 커널·드라이버는 DLAMI가 고정한 그대로 두고, 컨테이너만 더 최신 CUDA로 올려도 GPU 연산·NCCL·EFA 통신·학습이 정상 동작합니다.
- 컨테이너 시작 시 로그에 `CUDA Forward Compatibility mode ENABLED` 가 표시되면 이 모드로 동작 중이라는 뜻이며, 이는 오류가 아니라 정상 상태입니다.

이 메커니즘 덕분에 **"커널·드라이버는 안정적으로 고정, 프레임워크는 컨테이너로 자유롭게 최신화"** 라는 레이어 분리가 성립합니다. 커널을 올리지 않아도 되는 이유이기도 합니다.

!!! note "검증 환경 (2026-10 기준)"
    - 호스트 드라이버 CUDA: 13.2
    - 컨테이너(NGC `nvcr.io/nvidia/pytorch:26.08-py3`) CUDA: 13.4.1
    - 결과: Forward Compatibility mode로 GPU 연산, NCCL·EFA 통신까지 정상 동작 확인.

    사용자 학습 코드에 자체 CUDA 커널이 포함된 경우, Forward Compatibility 환경에서 빌드·동작 여부를 별도로 확인하는 것을 권장합니다. 드라이버 CUDA보다 높은 CUDA 툴킷으로 컴파일된 커스텀 커널은 이 모드의 지원 범위를 확인해야 합니다.

## 호스트 검증

인스턴스를 기동한 뒤, **각 노드에서** 아래 명령으로 사전 설치 구성요소가 모두 올바른 상태인지 확인합니다. 이 검증은 Slurm·컨테이너를 올리기 전에 수행해, 문제가 호스트 레이어에 있는지 상위 레이어에 있는지를 먼저 가려내는 단계입니다.

```bash
nvidia-smi --query-gpu=index,name,driver_version --format=csv   # GPU 인식·드라이버
systemctl is-active nvidia-fabricmanager                        # Fabric Manager 상태
modinfo efa | grep -E "^version"                                # EFA 커널 드라이버 버전
cat /opt/amazon/efa_installed_packages | head -3                # EFA installer 구성
ls /sys/class/infiniband/ | grep -c rdmap                       # EFA 장치 수
fi_info -p efa -t FI_EP_RDM | grep domain | sort -u             # libfabric이 보는 EFA 도메인
lsmod | grep gdrdrv; ls -l /dev/gdrdrv                           # GDRCopy
ulimit -l                                                       # memlock(핀 메모리 한도)
df -h /opt/dlami/nvme                                            # 로컬 NVMe 마운트
```

!!! note "FSx for Lustre를 공유 스토리지로 쓰는 경우"
    공유 스토리지로 FSx for Lustre를 사용한다면, Lustre client도 함께 검증합니다. (S3·FSx for OpenZFS 등 다른 스토리지를 쓰면 이 검증은 불필요합니다.)

    ```bash
    lfs --version   # Lustre client 설치·버전 확인
    ```

    GPU가 스토리지에서 직접 데이터를 읽는 **GPUDirect Storage(GDS)** 를 쓰려면 추가 점검이 필요합니다. [GDS (GPUDirect Storage)](gds-gpudirect-storage.md) 가이드를 참고하십시오.

### 각 명령이 확인하는 것

- **`nvidia-smi`** — GPU가 모두 인식되고 드라이버가 로드되었는지. 인스턴스 타입에 맞는 GPU 수와 모델이 보여야 합니다.
- **`nvidia-fabricmanager`** — NVLink/NVSwitch 패브릭을 관리하는 서비스. `active` 가 아니면 멀티‑GPU 집합 통신(NVLink 경로)이 성립하지 않습니다.
- **`modinfo efa`** — EFA 커널 드라이버가 설치되어 있는지와 그 버전. 노드 간 고속 통신의 기반입니다.
- **`efa_installed_packages`** — EFA installer가 함께 깐 libfabric·aws-ofi-nccl 등의 버전을 한눈에 확인합니다.
- **`/sys/class/infiniband/` 의 `rdmap*` 개수** 와 **`fi_info` 의 domain 수** — 실제로 노출된 EFA 장치 수를 OS 레벨과 libfabric 레벨 양쪽에서 확인합니다. 두 값이 인스턴스의 EFA 장치 수와 일치해야 합니다.
- **`gdrdrv` / `/dev/gdrdrv`** — GPUDirect RDMA 경로의 GDRCopy 커널 모듈이 로드되어 디바이스 노드가 존재하는지.
- **`ulimit -l` (memlock)** — 집합 통신 라이브러리는 핀(pinned) 메모리를 크게 잡으므로 `unlimited` 이어야 합니다. 제한되어 있으면 통신 초기화나 성능에 문제가 생깁니다.
- **`/opt/dlami/nvme`** — 로컬 NVMe가 LVM으로 묶여 자동 마운트되었는지. 컨테이너 이미지 캐시·스크래치 용도로 쓰입니다.
- **(FSx for Lustre 사용 시) `lfs --version`** — Lustre client가 설치되어 공유 스토리지 마운트가 가능한지.

### 기대 결과

| 항목 | 기대 결과 |
|---|---|
| GPU | 인스턴스 타입에 맞는 GPU가 전부 인식 |
| Fabric Manager | `active` |
| EFA 드라이버 | `modinfo efa` 에 버전이 표시 |
| EFA 장치 수 | `rdmap*` 개수 = `fi_info` domain 개수 = 인스턴스의 EFA 수 |
| GDRCopy | `gdrdrv` 로드, `/dev/gdrdrv` 존재 |
| memlock | `unlimited` |
| 로컬 NVMe | `/opt/dlami/nvme` 마운트됨 |

!!! note "검증 환경 (2026-10 기준)"
    P6-B200.48xlarge(인스턴스당 GPU 8, EFA 8)에서 측정한 기준값입니다.

    | 항목 | 기대 결과 |
    |---|---|
    | GPU | NVIDIA B200 × 8 |
    | Fabric Manager | `active` |
    | EFA 드라이버 | 3.3.0 이상 |
    | EFA 장치 수 | `rdmap*` 8개, `fi_info` domain 8개 |
    | GDRCopy | `gdrdrv` 로드, `/dev/gdrdrv` 존재 |
    | memlock | `unlimited` |

### ConnectX-7 장치가 함께 보이는 경우

일부 인스턴스 타입에서는 `/sys/class/infiniband/` 에 EFA(`rdmap*`) 외에 `ibp115s0f0`, `ibp116s0f0` 같은 **ConnectX-7 장치가 링크 DOWN 상태로 함께 보일 수 있습니다.** 이는 정상이며 EFA 통신에는 영향이 없습니다.

- NCCL은 EFA를 선택합니다(통신 로그에 `Using network Libfabric`, `provider is efa`, `found N nics` 표시).
- 따라서 별도 조치가 필요 없습니다. **`NCCL_NET=IB` 등으로 IB 경로를 강제하지 마십시오** — 링크 DOWN인 ConnectX-7을 억지로 쓰게 되어 통신 초기화가 실패합니다.

### 1노드 NCCL 확인 (NVLink만 사용)

호스트 레벨에서 GPU·드라이버·NCCL이 정상인지 노드 내부에서 먼저 확인합니다. (스케줄러 설치 전, 호스트만으로 수행하는 점검입니다.) 1노드 테스트는 EFA를 거치지 않고 노드 안 NVLink만 사용하므로, 노드 간 네트워크 문제와 GPU/드라이버 문제를 분리하는 데 유용합니다.

```bash
TESTDIR=/usr/local/cuda-13.2/efa/test-cuda-13.2
/opt/amazon/openmpi/bin/mpirun -n 8 --bind-to none \
  -x LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib:/usr/local/cuda-13.2/lib64:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH \
  -x NCCL_DEBUG=INFO \
  $TESTDIR/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 50
```

결과 로그에서 다음을 확인합니다.

- `Out of bounds values : 0 OK` — 계산 정확성. 0이 아니면 성능과 무관하게 실패입니다.
- busbw 값 — GPU·NVLink가 정상 대역을 내는지. 1노드 값은 이후 다노드(EFA) 테스트의 기준점이 됩니다.

!!! note "검증 환경 (2026-10 기준)"
    - nccl-tests 경로: `/usr/local/cuda-13.2/efa/test-cuda-13.2`
    - P6-B200 1노드 all-reduce(1 GiB) 참고값: **busbw 711 GB/s**

    nccl-tests 경로의 CUDA 버전 부분(`cuda-13.2`)은 AMI에 포함된 CUDA 버전에 맞춰 바뀝니다. 실제 환경의 `/usr/local/` 아래 경로를 확인해 사용하십시오. 위 성능 수치는 특정 시점·구성에서 측정한 참고값이며 성능을 보장하는 값이 아닙니다.

## 요약

- 호스트 레이어는 **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)** 를 기준으로 삼습니다. 드라이버·Fabric Manager·EFA·GDRCopy·Container Toolkit가 서로 버전이 맞춰진 채 묶여 있습니다(FSx for Lustre를 쓰면 Lustre client도 포함).
- **커널은 올리지 않습니다.** 커널과 커널 모듈(EFA 드라이버, 그리고 FSx for Lustre 사용 시 Lustre client)이 hold되어 있으며, 커널 변경은 이 정합을 깨뜨립니다. 운영용 AMI를 구울 때도 커널을 고정합니다.
- 컨테이너 CUDA가 호스트 드라이버 CUDA보다 높아도 **CUDA Forward Compatibility** 로 정상 동작합니다. 이것이 "커널은 고정, 프레임워크는 컨테이너로 최신화"를 가능케 합니다.
- 각 노드에서 `nvidia-smi` · Fabric Manager · EFA(드라이버·장치 수·libfabric) · GDRCopy · memlock을 **반드시 검증한 뒤** 상위 레이어로 넘어갑니다. (FSx for Lustre를 쓰면 Lustre client도 함께 확인합니다.)
