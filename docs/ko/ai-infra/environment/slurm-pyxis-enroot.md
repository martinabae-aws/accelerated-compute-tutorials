---
title: Slurm + Pyxis + Enroot
description: GPU 학습 클러스터의 스케줄러·컨테이너 런타임 레이어를 구성합니다. Slurm·munge 설치, 노드 이름 해석(IPv6 링크로컬 함정), Enroot·Pyxis 설정, 그리고 컨테이너 학습 잡 제출 방법을 다룹니다. 잡 제출과 nccl-tests 기반 분산 학습 검증(go/no-go) 포함.
---

# Slurm + Pyxis + Enroot

이 페이지는 GPU 학습 클러스터의 **스케줄러·컨테이너 런타임 레이어** 를 구성하는 방법을 설명합니다. 호스트(드라이버·EFA·컨테이너 툴킷 등)가 이미 검증된 상태를 전제로, 그 위에 다음을 올립니다.

- **Slurm** — 노드를 묶어 잡을 스케줄링하는 워크로드 매니저
- **munge** — 노드 간 인증(공유 키 기반)
- **Enroot** — 루트 권한 없이 컨테이너 이미지를 실행하는 비특권 런타임
- **Pyxis** — Slurm의 SPANK 플러그인으로, `srun` 한 줄에서 Enroot 컨테이너를 띄우게 해 주는 접착제

목표는 **노드 간 SSH 없이** `srun --container-image=...` 한 줄로 컨테이너 학습 잡을 전 노드에 띄우는 것입니다. 이 레이어에는 분산 학습 성능을 조용히 깎아먹거나 잡을 즉시 실패시키는 함정이 여럿 있어, 아래에서 "왜 이렇게 하는가"를 함께 설명합니다.

!!! note "검증 환경 (2026-10 기준)"
    아래 버전·값은 특정 시점에 검증한 레퍼런스 스냅샷입니다. 실제 구성에 맞게 갱신해 사용하십시오. 원칙(본문)은 버전과 무관하게 유효합니다.

    - 호스트 OS: Ubuntu 24.04 (Deep Learning Base OSS Nvidia Driver GPU AMI)
    - Slurm: 23.11.4 (Ubuntu apt 패키지)
    - Pyxis: 0.24.0
    - Enroot: 4.2.1
    - 인스턴스: GPU 노드(노드당 GPU 8개, EFA 8개)
    - 스케줄러 토폴로지: GPU 노드 1대가 컨트롤러(`slurmctld`)를 겸임
    - 노드 접속: SSM Session Manager (노드 간 SSH 없음)

## 아키텍처: 컨트롤러 겸임 구성

소규모~중규모 클러스터에서는 **GPU 노드 1대가 Slurm 컨트롤러(`slurmctld`)를 겸임** 하는 구성이 간단하고 실용적입니다. 별도 관리 노드를 두지 않아도 되고, 모든 노드가 동일한 AMI·동일한 데몬 집합을 쓰므로 운영이 단순합니다.

```
[ GPU 노드 × N (한 블록 권장) ]
  노드 1 : slurmctld(컨트롤러 겸임) + slurmd + munge + Enroot + Pyxis
  노드 2 : slurmd + munge + Enroot + Pyxis
  ...
  노드 N : slurmd + munge + Enroot + Pyxis

  네트워크 : 노드 안 NVLink / 노드 간 EFA
  공유 스토리지 : /fsx (컨테이너 이미지 sqsh, 데이터, 체크포인트)
```

- 모든 노드에 `slurmd`가 돌고, 그중 1대가 추가로 `slurmctld`를 실행합니다.
- 노드 간 인증은 munge 공유 키로 처리하므로 노드 간 SSH가 필요 없습니다. 분산 실행은 전부 `srun`을 통해 이뤄집니다.
- Slurm·Pyxis·Enroot를 설치해 둔 상태를 하나의 **운영용 AMI** 로 구워 두면, 장애 노드 교체 시 같은 템플릿으로 다시 띄우기만 하면 됩니다.

!!! note "검증 환경 (2026-10 기준)"
    검증 구성은 Ubuntu apt의 Slurm 23.11.4를 사용했습니다. 실제 환경에서 Slurm을 직접 빌드하거나 다른 패키지로 설치하는 경우에도, 아래 각 항목(노드 이름 해석, munge, Enroot `/run` 디렉터리, Pyxis 플러그인 등록)은 동일하게 적용됩니다.

## 1단계. 노드 이름 해석 (가장 흔한 함정)

Slurm은 컨트롤러와 각 `slurmd`가 서로를 **IPv4로** 찾아 통신합니다. 그런데 최신 Ubuntu(24.04)에서는 **노드의 자기 호스트 이름이 IPv6 링크 로컬 주소(`fe80::...`)로 해석될 수 있습니다.** 이 상태에서는 Slurm 노드가 `down`으로 빠지거나 통신 오류가 납니다.

먼저 현재 해석 결과를 확인합니다.

```bash
getent ahostsv4 $(hostname)   # IPv4로 풀리는지
getent hosts $(hostname)      # fe80::... (IPv6 링크로컬)로 풀리면 문제
```

`fe80::...`가 보이면 다음 중 **하나** 를 적용해 호스트 이름이 반드시 IPv4로 풀리도록 고정합니다.

- `slurm.conf`의 노드 정의에 `NodeAddr=<사설 IPv4>`를 명시한다. (각 노드의 통신 주소를 Slurm에 직접 알려 주는 방법)
- 모든 노드의 `/etc/hosts`에 `<사설 IPv4>  <호스트 이름>` 항목을 추가한다.

두 방법 모두 "Slurm이 노드를 IPv4로 찾게 한다"는 같은 목적을 달성합니다. 운영 AMI를 구울 때 이 설정을 포함시켜 두면 노드 교체 때마다 다시 손댈 필요가 없습니다.

## 2단계. munge (노드 간 인증)

munge는 Slurm 데몬 간 요청을 인증하는 공유 키 메커니즘입니다. **모든 노드가 동일한 `/etc/munge/munge.key`를 가져야** 하며, 키 파일은 소유자 `munge`, 권한 `400`이어야 합니다.

```bash
# (컨트롤러에서 한 번) 키 생성
sudo /usr/sbin/mungekey --verbose
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key
```

노드 간 SSH가 없으므로, 생성한 키는 **공유 스토리지(`/fsx`)를 경유해** 모든 노드에 배포한 뒤 각 노드의 `/etc/munge/`로 복사합니다. 배포가 끝나면 공유 위치에 둔 사본은 반드시 삭제하십시오(키 노출 방지).

```bash
# 각 노드에서
sudo install -o munge -g munge -m 400 /fsx/<path>/munge.key /etc/munge/munge.key
sudo systemctl enable --now munge
munge -n | unmunge | grep STATUS   # STATUS: Success (0) 확인
```

## 3단계. slurm.conf 와 gres.conf

Slurm의 노드·리소스 정의입니다. 핵심은 (1) GPU를 GRES(Generic RESource)로 등록하고, (2) 노드의 하드웨어 토폴로지를 정확히 기술하는 것입니다. 노드 값은 **각 노드에서 `slurmd -C`로 출력되는 값을 그대로** 써야 합니다(틀리면 노드가 `down`됩니다).

```ini
SlurmctldHost=<controller>(<controller IPv4>)
AuthType=auth/munge
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
SelectType=select/cons_tres
SelectTypeParameters=CR_Core
GresTypes=gpu

# 노드 공통 기본값 — slurmd -C 출력에 맞춤
NodeName=DEFAULT CPUs=192 Boards=1 SocketsPerBoard=2 CoresPerSocket=48 \
  ThreadsPerCore=2 RealMemory=<slurmd -C 값보다 약간 작게> Gres=gpu:8 State=UNKNOWN

# 노드별 통신 주소 명시 (1단계의 IPv6 함정 회피)
NodeName=<host-1> NodeAddr=<IPv4-1>
NodeName=<host-2> NodeAddr=<IPv4-2>
# ...

PartitionName=train Nodes=ALL Default=YES MaxTime=INFINITE State=UP
```

`gres.conf`에서 GPU 디바이스 파일을 등록합니다.

```ini
Name=gpu File=/dev/nvidia[0-7]
```

지킬 것:

- **`RealMemory`는 `slurmd -C`가 보고하는 값보다 약간 작게** 잡습니다. 실제 가용 메모리가 보고치보다 조금이라도 작으면 노드가 `down`으로 빠지기 때문입니다.
- `cgroup.conf`의 `ConstrainDevices`를 `yes`로 운영하면 잡에 할당된 GPU만 보이도록 격리됩니다. 이 경우 잡 안에서 `/dev/infiniband`(EFA)와 `/dev/gdrdrv`(GDRCopy)가 **여전히 보이는지** 반드시 확인하십시오. 이들이 가려지면 EFA 통신·GPUDirect가 동작하지 않습니다.

!!! note "검증 환경 (2026-10 기준)"
    - 노드 토폴로지: `CPUs=192`, `SocketsPerBoard=2`, `CoresPerSocket=48`, `ThreadsPerCore=2`, `Gres=gpu:8` (`slurmd -C` 출력 기준)
    - 검증 구성은 `cgroup.conf`에 `ConstrainDevices=no`로 두었습니다. `yes`로 운영하려면 위 디바이스 가시성 확인이 필요합니다.

## 4단계. PMIx (MPI 런처)

노드 간 SSH 없이 MPI 기반 프로그램(예: `nccl-tests`의 `all_reduce_perf`)을 띄우려면 Slurm의 `--mpi=pmix`를 사용합니다. `srun`이 PMIx로 각 랭크를 조율하므로 호스트 간 SSH가 필요 없습니다.

```bash
srun --mpi=list   # 목록에 pmix 가 있어야 --mpi=pmix 사용 가능
```

- `--mpi=pmix`는 Slurm이 PMIx 지원으로 빌드된 경우에만 사용할 수 있습니다. Slurm을 직접 빌드한다면 `--with-pmix`로 빌드하십시오.

!!! note "검증 환경 (2026-10 기준)"
    Ubuntu apt의 Slurm 23.11.4는 PMIx(v5) 지원으로 빌드되어 있어 `srun --mpi=list`에 `pmix`가 나옵니다.

## 5단계. Enroot (비특권 컨테이너 런타임)

Enroot는 루트 권한 없이 컨테이너를 실행합니다. 설정의 핵심은 (1) 런타임 디렉터리 `/run/enroot`를 **매 부팅마다 보장** 하고, (2) 캐시·데이터처럼 용량이 큰 경로를 **노드 로컬 NVMe** 에 두는 것입니다.

```bash
sudo tee /etc/enroot/enroot.conf > /dev/null <<'CONF'
ENROOT_RUNTIME_PATH      /run/enroot/user-$(id -u)
ENROOT_CACHE_PATH        /opt/dlami/nvme/enroot/cache/user-$(id -u)
ENROOT_DATA_PATH         /opt/dlami/nvme/enroot/data/user-$(id -u)
ENROOT_TEMP_PATH         /opt/dlami/nvme/enroot/tmp
ENROOT_SQUASH_OPTIONS    -noI -noD -noF -noX -no-duplicates
ENROOT_MOUNT_HOME        y
ENROOT_RESTRICT_DEV      n
CONF

sudo mkdir -p /opt/dlami/nvme/enroot/tmp
sudo chmod 1777 /opt/dlami/nvme/enroot /opt/dlami/nvme/enroot/tmp
```

### `/run/enroot` 를 tmpfiles로 보장하기

`/run`은 tmpfs이므로 **재부팅 시 비워집니다.** 이 디렉터리가 없으면 Pyxis 잡이 다음과 같이 실패합니다.

```
mkdir: cannot create directory '/run/enroot': Permission denied
```

따라서 systemd-tmpfiles로 등록해 **부팅 때마다 자동 생성** 되게 합니다. (잊고 수동으로 한 번만 만들면, 노드 재부팅 후 다시 깨집니다.)

```bash
echo "d /run/enroot 1777 root root -" | sudo tee /etc/tmpfiles.d/enroot.conf
sudo systemd-tmpfiles --create /etc/tmpfiles.d/enroot.conf
```

지킬 것:

- **캐시·데이터는 노드 로컬 NVMe(`/opt/dlami/nvme`)에 둡니다.** 컨테이너 이미지는 수십 GB에 달하므로 루트 EBS에 두면 금방 가득 찹니다.
- **`ENROOT_RESTRICT_DEV n`** 으로 두면 컨테이너 안에서 `/dev/infiniband`(EFA)가 별도 마운트 지정 없이 그대로 보입니다. EFA 통신에 필요합니다.

## 6단계. Pyxis (Slurm ↔ Enroot 접착제)

Pyxis는 Slurm의 SPANK 플러그인으로 설치됩니다. 등록되면 `srun`에 `--container-image` 같은 옵션이 생기고, 잡이 뜰 때 Pyxis가 자동으로 Enroot 컨테이너를 띄웁니다.

```bash
# 빌드에는 Slurm 헤더(libslurm-dev)가 필요합니다.
git clone https://github.com/NVIDIA/pyxis.git && cd pyxis && sudo make install

# SPANK 플러그인 등록
echo "required /usr/local/lib/slurm/spank_pyxis.so" | sudo tee /etc/slurm/plugstack.conf.d/pyxis.conf
echo "include /etc/slurm/plugstack.conf.d/*"        | sudo tee /etc/slurm/plugstack.conf

sudo systemctl restart slurmd
```

등록 확인:

```bash
srun --help | grep container-image   # --container-image 옵션이 나오면 성공
```

## 7단계. 기본 동작 확인

여기까지 왔으면 스케줄러·컨테이너 레이어가 완성됩니다. 아래 순서로 점검합니다.

```bash
sinfo                                   # 노드가 idle 상태로 보이는지
srun -N2 --ntasks-per-node=1 hostname   # 노드 간 실행 (munge·PMIx 경로)

# Pyxis + Enroot 컨테이너 실행 (NGC 이미지)
srun -N2 --ntasks-per-node=1 --gres=gpu:8 \
  --container-image=nvcr.io#nvidia/pytorch:<tag> \
  bash -c 'hostname; nvidia-smi -L | wc -l; ls /dev/infiniband | head -3'
```

- `nvidia-smi -L`이 노드당 8줄, `/dev/infiniband`에 EFA 장치가 보이면 컨테이너가 GPU·EFA를 정상적으로 집어낸 것입니다.
- `--container-image`의 레지스트리 표기는 `nvcr.io#nvidia/pytorch:<tag>` 형식(Pyxis 규약)입니다. 공유 스토리지의 sqsh 파일을 쓸 때는 `--container-image=/fsx/<path>/<image>.sqsh`로 지정합니다.

!!! note "검증 환경 (2026-10 기준)"
    컨테이너 이미지는 `nvcr.io/nvidia/pytorch:26.08-py3`(및 여기에 EFA installer 1.50.0을 추가한 이미지)로 검증했습니다. NGC 이미지에는 AWS EFA 스택이 이미 포함되어 있어, 2노드 컨테이너 통신에서 호스트와 동일한 성능이 나왔습니다(단, `--cpus-per-task` 지정 필수 — 아래 참고).

> 📎 [NVIDIA Pyxis](https://github.com/NVIDIA/pyxis) · [NVIDIA Enroot](https://github.com/NVIDIA/enroot)

## 잡 제출

스케줄러·컨테이너 레이어가 서면, 실제 학습 잡은 `sbatch`/`salloc` + `srun`으로 제출합니다. 아래는 컨테이너 학습 잡의 기본형입니다.

```bash
#!/bin/bash
#SBATCH -N <노드 수>
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500

srun --container-image=/fsx/<path>/<image>.sqsh \
     --container-mounts=/fsx:/fsx \
     <학습 실행 스크립트>
```

구조:

- 노드당 GPU 8개에 **8 rank**(`--ntasks-per-node=8`)를 띄우고, 각 rank가 GPU 1개를 씁니다.
- `--gres=gpu:8` 로 노드당 GPU 8장을 요청합니다.
- 마스터 노드 주소는 연속 번호를 가정하지 말고 **`scontrol show hostnames "$SLURM_JOB_NODELIST"`** 로 받습니다.

### 반드시 지킬 두 가지 옵션

이 두 옵션은 빠뜨려도 잡이 "뜨긴" 하기 때문에 특히 놓치기 쉽습니다. 하나는 성능을 조용히 깎고, 하나는 잡을 즉시 죽입니다.

**1. `--cpus-per-task` 를 반드시 지정** 

rank당 할당할 CPU 코어 수입니다. 지정하지 않으면 Slurm 설정에 따라 rank당 **CPU 1코어만** 배정될 수 있고, 이 경우 통신 성능이 **기준값의 1/4~1/10 수준으로** 떨어집니다. EFA는 멀쩡히 "타는데"(`Using network Libfabric`) 대역폭만 낮게 나오는, 원인을 찾기 어려운 증상입니다. 네트워크 스택이 패킷을 처리할 CPU가 부족하기 때문입니다.

**2. `srun --export` 를 쓸 때는 `ALL` 을 함께** 

`srun --export`로 환경변수를 전달할 때는 반드시 `ALL`을 포함하십시오(`--export=ALL,VAR=value,...`). `ALL` 없이 변수만 나열하면 `PATH` 같은 기본 환경이 전달되지 않아, 잡이 다음과 같이 **즉시 실패** 합니다.

```
slurmstepd: error: execve(): bash: No such file or directory
```

보조 규칙:

- 잡 스크립트 안에서 `export`로 설정하고 `--export`를 생략해도 됩니다(기본값이 `ALL`).
- `--export` 값 안에는 **쉼표를 쓸 수 없습니다**(Slurm이 변수 구분자로 읽음). 예를 들어 `NCCL_DEBUG_SUBSYS=INIT,NET`처럼 쉼표가 든 값은 `--export`에 넣지 말고 잡 스크립트에서 `export`로 설정하십시오.
- 한 할당(`salloc`/`sbatch`) 안에서 여러 `srun`을 실행할 때는 `--container-name=<이름>`으로 컨테이너를 재사용하면, 두 번째부터는 sqsh를 다시 풀지 않고 바로 시작합니다.

### 환경변수 (EFA + NCCL)

NGC 계열 이미지에는 AWS EFA 스택과 올바른 기본 환경변수가 이미 들어 있습니다. **대부분의 변수는 지정하지 않는 것이 정답** 이며, 잘못 지정하면 TCP로 fallback하거나 초기화가 실패합니다.

| 변수 | 필요 여부 | 설명 |
|---|---|---|
| `NCCL_NET` | **지정 불필요** | 이미지 기본값(`NCCL_NET_PLUGIN=ofi`)이 EFA 플러그인을 고릅니다. 굳이 지정하면 `Libfabric`(대소문자 무관)이어야 하고, `IB`를 지정하면 초기화가 실패, 이전 이름(`AWS Libfabric`)은 통신 결과가 나오지 않습니다. 두 값 모두 쓰지 마십시오. |
| `NCCL_NET_PLUGIN` | 바꾸지 말 것 | 이미지 기본값 `ofi` 유지. 다른 값으로 바꾸면 IB용 플러그인을 가리켜 EFA 경로를 벗어납니다. |
| `FI_PROVIDER`, `FI_EFA_USE_DEVICE_RDMA` | 지정 불필요 | 지정하지 않아도 EFA·GPUDirect RDMA가 정상 사용됩니다. 지정해도 무방. |
| `NCCL_DEBUG=INFO` | 점검 시 | 잡 시작 로그에서 `Using network Libfabric`, `found 8 nics`를 확인할 때 켭니다. |
| IB 환경의 `NCCL_IB_*`, `NCCL_NET=IB` 등 | **제거** | EFA 환경에서는 쓰지 않습니다. IB용 설정이 남아 있으면 EFA 경로를 벗어나 TCP로 떨어지거나 초기화가 실패합니다. |

- 핵심 원칙: **EFA 환경에서는 NCCL 네트워크 관련 변수를 "건드리지 않는 것"이 기본** 입니다. 다른 환경(IB)에서 가져온 `NCCL_*` 설정을 그대로 쓰면 성능이 떨어지거나 잡이 실패합니다.
- 점검 시 `NCCL_DEBUG=INFO` 로그에서 `Using network Libfabric`과 `provider is efa (... found 8 nics)`가 보이면 EFA 경로입니다. `Using network Socket`이 보이면 TCP로 fallback한 것이므로(성능 급락) `NCCL_NET_PLUGIN=ofi` 여부와 잔존 IB 설정을 점검하십시오.

!!! note "검증 환경 (2026-10 기준)"
    - aws-ofi-nccl 1.19.0(NGC 26.08 내장)과 1.21.1(EFA installer 1.50.0)에서 동작을 확인했으며 **1.21.1을 권장** 합니다.
    - `--cpus-per-task` 미지정 시 성능 하락 실측(8 GiB): all-reduce algbw 92 → 368 GB/s, all-to-all 8.7 → 95 GB/s (`--cpus-per-task=24` 적용 시 복원).
    - 위 두 버전 모두 `NCCL_NET`을 지정하지 않아도 EFA 플러그인이 선택됩니다.

> 📎 [Pyxis 사용법](https://github.com/NVIDIA/pyxis/blob/main/README.md) · [NCCL 환경변수](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

## 트러블슈팅 요약

이 레이어에서 실제로 자주 마주치는 증상과 조치입니다.

| 증상 | 원인 | 조치 |
|---|---|---|
| Slurm 노드가 `down`, 통신 오류, `getent hosts $(hostname)`이 `fe80::...` | 자기 호스트 이름이 IPv6 링크 로컬로 풀림 | `slurm.conf`에 `NodeAddr`(IPv4) 명시 또는 `/etc/hosts` 등록 (1단계) |
| 노드가 `down`, `RealMemory` 관련 | 실제 가용 메모리 < 설정값 | `RealMemory`를 `slurmd -C` 값보다 약간 작게 (3단계) |
| Pyxis 잡이 `mkdir: cannot create directory '/run/enroot': Permission denied` | 재부팅으로 `/run/enroot` 사라짐 | tmpfiles로 생성 보장 (5단계) |
| `slurmstepd: error: execve(): bash: No such file or directory` | `--export`에 `ALL` 없음 | `--export=ALL,...` (잡 제출) |
| EFA는 타는데(`Using network Libfabric`) 통신이 기준값의 1/4~1/10 | rank당 CPU 1코어 | `--cpus-per-task=24` (잡 제출) |
| NCCL 로그에 `Using network Socket` | EFA 플러그인 미로드 / TCP fallback | `NCCL_NET_PLUGIN`이 `ofi`인지, IB용 `NCCL_*` 잔존 여부 확인 |
| `NCCL_NET=IB` 등으로 통신 초기화 실패 | EFA 환경에 IB 경로 강제 | IB 설정 제거 (잡 제출) |
| 컨테이너 안에서 `/dev/infiniband`가 안 보임 | `cgroup.conf` 디바이스 격리 또는 `ENROOT_RESTRICT_DEV` 설정 | `ConstrainDevices` 시 디바이스 가시성 확인(3단계), `ENROOT_RESTRICT_DEV n`(5단계) |

## 분산 학습 검증 (nccl-tests) { #distributed-training-validation }

클러스터의 노드·네트워크·컨테이너 구성을 모두 마쳤다면, 실제 학습을 시작하기 **전에** 노드 간 집합 통신(collective communication) 성능이 기대 수준으로 나오는지 반드시 확인해야 합니다. 아래는 그 **최종 검증 게이트** 입니다. 여기서 통과하지 못하면(no-go) 학습을 시작해도 네트워크 병목으로 성능이 나오지 않거나 학습이 비정상적으로 느려집니다.

이 단계에서 수행하는 것:

- 2노드 all-reduce / all-to-all 대역폭 측정 (호스트 nccl-tests)
- 노드 간 지연 측정 (`fi_pingpong`)
- 컨테이너(학습 이미지) 안에서의 통신 측정 (PyTorch 기반, MPI 충돌 회피)
- nccl-tests 로그를 읽고 EFA 경로 여부·성능을 판정하는 방법

### 왜 이 게이트가 필요한가

대규모 GPU 학습에서 각 스텝은 **GPU 연산 + 노드 간 통신** 으로 이루어집니다. 노드 안(GPU 8개)은 NVLink로, 노드 사이는 EFA로 연결됩니다. 노드 간 EFA 대역폭은 인스턴스당 약 3.2 Tbps(약 400 GB/s), GPU 8개로 나누면 **GPU당 약 50 GB/s** 로, NVLink(GPU당 한 방향 수백 GB/s)와 큰 차이가 납니다. 즉 노드 수가 늘수록 노드 간 통신이 전체 처리량을 좌우하므로, 그 경로가 **설계 대역폭에 가깝게 동작하는지** 를 학습 전에 확인해야 합니다.

이 검증이 중요한 또 다른 이유는, 통신 성능 저하의 원인 대부분이 **학습을 돌려 보기 전에는 드러나지 않는 설정 문제** 이기 때문입니다. 대표적으로:

- NCCL이 EFA 플러그인을 못 잡아 TCP(Socket)로 fallback → 성능이 수분의 1로 떨어짐
- rank당 CPU가 1코어만 배정되어 통신 성능이 1/4~1/10로 저하
- 보안그룹 self-reference 규칙 누락으로 EFA 통신 자체가 실패
- IB용 환경변수가 남아 있어 EFA가 선택되지 않음

이 게이트에서 **기준선(baseline)** 수치를 확보해 두면, 이후 운영 중 성능이 떨어졌을 때 "원래 이 환경에서 얼마가 나왔는지"와 비교할 수 있습니다. 그래서 측정값은 반드시 기록해 두어야 합니다.

### 검증 순서 (권장)

통신 경로를 아래 → 위 순서로 하나씩 좁혀 가며 검증합니다. 아래 단계에서 문제가 있으면 위 단계는 의미가 없으므로, 반드시 순서대로 진행합니다.

1. **호스트 EFA·NCCL 기준선** — DLAMI 내장 nccl-tests로 2노드 이상 all-reduce·all-to-all 측정 (Slurm `srun --mpi=pmix`)
2. **컨테이너 NCCL 경로** — 학습 이미지로 PyTorch 측정 스크립트를 실행해 호스트와 비슷한지 확인 (`--cpus-per-task` 필수)
3. **노드 간 지연** — `fi_pingpong`으로 소형 메시지 왕복 지연 측정
4. **(선택) EP 디스패처 확인** — MoE를 쓰는 경우 Expert Parallelism 백엔드 단독 테스트
5. **대상 모델·병렬화 구성으로 짧은 학습** — 몇십 iteration만 돌려 iteration 시간을 기준선으로 기록
6. **(용량 블록이 여러 개로 나뉘었다면)** 토폴로지를 확인한 뒤 블록 안·블록 사이를 각각 측정해 비교

아래는 1~4단계(순수 통신 검증)에 집중합니다.

### 1. 호스트 2노드 nccl-tests (EFA 기준선)

가장 먼저 **컨테이너 없이 호스트에서** nccl-tests를 돌려 EFA·NCCL 경로가 정상인지 확인합니다. 호스트에는 드라이버·EFA 스택·nccl-tests가 사전 설치되어 있어, 컨테이너 변수를 배제한 "순수한" 기준선을 얻을 수 있습니다.

- nccl-tests는 CUDA 버전별로 설치되어 있습니다. 노드당 8 rank, rank당 GPU 1개(`-g 1`)로 실행합니다.
- 노드 간 SSH 없이 `srun --mpi=pmix`로 실행합니다. 이는 Slurm이 PMIx 지원으로 빌드된 경우에만 가능합니다 — `srun --mpi=list`에 `pmix`가 보이는지 먼저 확인하십시오.
- **`--cpus-per-task`를 반드시 지정하십시오.** 지정하지 않으면 rank당 CPU 1코어만 배정될 수 있고, 이때 통신 성능이 크게 떨어집니다.

!!! note "검증 환경 (2026-10 기준)"
    아래 경로·버전·수치는 특정 시점에 P6-B200 2노드 환경에서 측정한 **참고 레퍼런스** 입니다. 버전이 바뀌면 이 박스만 갱신하면 됩니다.

    - DLAMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)
    - nccl-tests 경로: `/usr/local/cuda-XX/efa/test-cuda-XX/` (CUDA 12.8·12.9·13.0·13.2 각각 제공)
    - 드라이버 기준 CUDA 13.2, EFA installer 1.50.0 (libfabric 2.6.0, aws-ofi-nccl 1.21.1, PMIx 4.2.8)
    - Slurm 23.11.4 (PMIx v5 지원), Pyxis 0.24.0, Enroot 4.2.1
    - 아래 예시 명령의 `cuda-13.2`는 실제 설치된 CUDA 버전에 맞춰 바꾸십시오.

```bash
TESTDIR=/usr/local/cuda-13.2/efa/test-cuda-13.2
LIBS=/usr/local/cuda-13.2/lib:/usr/local/cuda-13.2/lib64:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib

# all-reduce (데이터 병렬 통신 패턴)
srun -N2 --ntasks-per-node=8 --cpus-per-task=24 --gres=gpu:8 --mpi=pmix \
  --export=ALL,LD_LIBRARY_PATH=$LIBS,NCCL_DEBUG=INFO \
  $TESTDIR/all_reduce_perf -b 8 -e 16G -f 2 -g 1 -c 1 -n 50

# all-to-all (MoE expert 병렬 통신 패턴)
srun -N2 --ntasks-per-node=8 --cpus-per-task=24 --gres=gpu:8 --mpi=pmix \
  --export=ALL,LD_LIBRARY_PATH=$LIBS,NCCL_DEBUG=INFO \
  $TESTDIR/alltoall_perf -b 8 -e 16G -f 2 -g 1 -c 1 -n 50
```

플래그 의미:

- `-b 8 -e 16G` : 메시지 크기를 8바이트부터 16 GiB까지 스윕
- `-f 2` : 크기를 2배씩 증가
- `-g 1` : rank당 GPU 1개 (rank는 `--ntasks-per-node=8`로 이미 노드당 8개)
- `-c 1` : 정확성 검사 수행
- `-n 50` : 각 크기당 50회 반복 측정

!!! tip "`--export`에는 반드시 `ALL`을 함께 쓰십시오"
    `srun --export`에 변수만 나열하면 `PATH` 등이 전달되지 않아 잡이 바로 실패할 수 있습니다. 반드시 `--export=ALL,VAR=value,...` 형태로 쓰거나, 잡 스크립트에서 `export`로 설정하고 `--export`를 생략하십시오(기본값이 `ALL`). 또한 `--export` 값 안에는 쉼표를 쓸 수 없습니다(Slurm이 변수 구분자로 읽음).

#### 1노드 사전 확인 (NVLink만)

2노드로 넘어가기 전에, 1노드 all-reduce로 GPU·드라이버·NCCL 자체가 정상인지 먼저 확인하면 문제 범위를 좁히는 데 도움이 됩니다. 1노드는 노드 안 NVLink만 사용하므로 EFA와 무관하게 GPU 스택의 건전성을 봅니다.

```bash
TESTDIR=/usr/local/cuda-13.2/efa/test-cuda-13.2
/opt/amazon/openmpi/bin/mpirun -n 8 --bind-to none \
  -x LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib:/usr/local/cuda-13.2/lib64:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH \
  -x NCCL_DEBUG=INFO \
  $TESTDIR/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 50
```

### 2. 컨테이너 2노드 통신 (PyTorch 측정)

호스트 기준선이 확인되면, **실제 학습에 쓸 컨테이너 이미지 안** 에서도 같은 성능이 나오는지 검증합니다. 이미지 안의 EFA 스택·NCCL·플러그인 설정이 호스트와 다를 수 있기 때문입니다.

!!! warning "컨테이너 안에서는 nccl-tests를 MPI로 돌리지 마십시오"
    학습용 NGC 계열 이미지에는 자체 Open MPI가 들어 있습니다. EFA installer의 Open MPI로 빌드한 프로그램(nccl-tests 포함)을 컨테이너 안에서 MPI로 실행하면 두 Open MPI가 섞여 `opal_shmem_base_select failed` → `MPI_INIT failed`로 실패합니다. 따라서 **컨테이너 안 통신 확인은 MPI가 필요 없는 PyTorch 기반 측정** 으로 수행합니다. 이 스크립트는 Slurm 환경변수(`SLURM_PROCID` 등)로 rank를 잡아 MPI 없이 분산 그룹을 초기화합니다.

측정 스크립트를 공유 스토리지에 두고, 학습 이미지로 실행합니다.

```bash
salloc -N2 --ntasks-per-node=8 --cpus-per-task=24 --gres=gpu:8
srun --container-image=/<shared-path>/<image>.sqsh --container-name=train --container-mounts=/<shared-path>:/<shared-path> \
  --export=ALL,MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1),MASTER_PORT=29500,NCCL_DEBUG=INFO \
  python /<shared-path>/torch_comm_bench.py
```

!!! info "참고 측정 스크립트 — `torch_comm_bench.py`"
    MPI 없이 Slurm 환경변수로 rank를 잡는 PyTorch 측정 스크립트입니다. busbw 계산식은 nccl-tests와 동일합니다. 특정 환경을 위한 참고 예시이므로, 라이브러리 버전·측정 범위는 실제 환경에 맞춰 조정하십시오.

    ```python
    import os, time, torch, torch.distributed as dist

    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world,
                            device_id=torch.device("cuda", local))

    sizes = [2**k for k in range(20, 34)]
    for op in ["all_reduce", "all_to_all"]:
        for nbytes in sizes:
            x = torch.ones(nbytes // 4, dtype=torch.float32, device="cuda")
            y = torch.empty_like(x)
            def run():
                if op == "all_reduce":
                    dist.all_reduce(x)
                else:
                    dist.all_to_all_single(y, x)
            for _ in range(5):
                run()
            torch.cuda.synchronize()
            dist.barrier()
            iters = 20
            t0 = time.time()
            for _ in range(iters):
                run()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / iters
            algbw = nbytes / dt / 1e9
            busbw = algbw * 2 * (world - 1) / world if op == "all_reduce" else algbw * (world - 1) / world
            if rank == 0:
                print(f"{op:10s} {nbytes:>12d} B  time {dt*1e3:9.2f} ms  "
                      f"algbw {algbw:7.1f} GB/s  busbw {busbw:7.1f} GB/s", flush=True)
            del x, y
    dist.destroy_process_group()
    ```

!!! warning "rank당 CPU를 충분히 배정하십시오"
    `--cpus-per-task`를 지정하지 않으면 rank당 CPU 1코어만 배정될 수 있고, 이때 EFA가 정상 동작해도 통신 성능이 **1/4~1/10로 떨어집니다**. 한 측정에서 8 GiB all-reduce가 algbw 92 → 368 GB/s, all-to-all이 8.7 → 95 GB/s로 바뀐 사례가 있습니다. 인스턴스의 vCPU를 GPU 수로 나눈 값(예: 192 vCPU ÷ 8 GPU = 24)을 지정하십시오.

### 3. 노드 간 지연 (`fi_pingpong`)

대역폭과 별개로, 소형 메시지의 **왕복 지연(latency)** 도 측정해 둡니다. all-reduce는 대역폭에 민감하지만, 작은 메시지가 많은 통신 패턴은 지연에 민감합니다.

```bash
salloc -N2 --ntasks-per-node=1
srun --overlap -N1 -n1 -w <node-A> /opt/amazon/efa/bin/fi_pingpong -p efa -e rdm -I 10000 -S 64 &
sleep 5
srun --overlap -N1 -n1 -w <node-B> /opt/amazon/efa/bin/fi_pingpong -p efa -e rdm -I 10000 -S 64 <node-A IP>
wait
```

- `fi_pingpong`의 기본 endpoint는 dgram이므로 `-e rdm`을 지정합니다.
- `-I`는 반복 횟수, `-S`는 메시지 크기(바이트)입니다.
- 서버 측을 먼저 띄우고(`&`), 잠깐 기다린 뒤 클라이언트 측에서 서버 IP를 지정해 접속합니다.

### 4. 기준값과 판정

측정이 끝나면 아래 기준값과 비교하고, **실 환경 측정값을 기준선으로 기록** 해 둡니다.

!!! note "검증 환경 (2026-10 기준) — 참고 기준값"
    아래 수치는 특정 시점에 P6-B200 **2노드** 환경에서 측정한 **참고값이며, 성능을 보장하는 값이 아닙니다**. 노드 수, 모델, 병렬화 구성, 소프트웨어 버전에 따라 달라지므로 실 환경 측정값을 기준선으로 삼으십시오.

    | 측정 | 참고 기준값 |
    |---|---|
    | 호스트 all-reduce (16 GiB) | algbw 374 GB/s, busbw 702 GB/s |
    | 호스트 all-to-all (16 GiB) | algbw 95 GB/s, busbw 89 GB/s |
    | 컨테이너 all-reduce / all-to-all (8 GiB, PyTorch) | algbw 367~368 / 95 GB/s |
    | `fi_pingpong` 64 B | 10.85 µs/xfer |

    - 참고로 노드당 EFA 대역폭은 약 400 GB/s이며, 위 all-reduce algbw 374 GB/s는 그 약 94% 수준입니다.
    - all-to-all algbw 95 GB/s는 GPU당 노드 간 트래픽으로 환산하면 약 48 GB/s로, GPU당 EFA 몫(약 50 GB/s)의 약 95% 수준입니다.

판정 지침:

- **노드를 늘려 측정할 때 값이 크게 떨어지면**, 노드를 절반씩 나눠 다시 측정해 문제 노드를 좁힙니다. 문제 노드는 종료 후 같은 런치 템플릿으로 다시 기동해 교체합니다.
- 용량 블록이 여러 개로 나뉘어 있으면 블록 안 노드 쌍과 블록 사이 노드 쌍을 각각 측정해 비교하십시오. 블록 사이는 네트워크 구간(spine)이 달라 성능이 다를 수 있습니다.

### 5. nccl-tests 결과 읽는 법

`NCCL_DEBUG=INFO`를 켜고 실행한 뒤, 아래 순서로 확인합니다.

#### ① 계산이 맞았는가 — 오류 수

```
# Out of bounds values : 0 OK
```

0이어야 합니다. 결과 표의 `#wrong` 열도 모두 0이어야 합니다. **0이 아니면 성능과 관계없이 실패** 입니다(no-go).

#### ② 어떤 네트워크를 탔는가 — 로그 앞부분

```
NET/OFI Initializing aws-ofi-nccl <버전>
NET/OFI Selected provider is efa, fabric is efa-direct (found 8 nics)
Using network Libfabric
```

- `Using network Libfabric`와 `provider is efa`가 보여야 EFA 경로입니다.
- `found 8 nics`(EFA 8개)가 보여야 모든 네트워크 카드를 쓰는 것입니다.
- `Using network Socket`이 보이면 TCP로 fallback한 것으로, 성능이 크게 떨어집니다. 이 경우 아래 트러블슈팅을 참고하십시오.
- GPUDirect RDMA 사용은 로그의 `via NET/Libfabric/.../GDRDMA`로 확인합니다.

#### ③ 성능 — 결과 표의 `algbw`·`busbw`

```
      size    count  type  redop root   time   algbw   busbw  #wrong | (in-place) ...
```

| 열 | 의미 |
|---|---|
| size | 한 번에 보내는 데이터 크기(바이트) |
| time | 걸린 시간(마이크로초) |
| algbw | 데이터 크기 ÷ 시간 (알고리즘 대역폭) |
| **busbw** | algbw를 통신 패턴에 맞게 보정한 값. 하드웨어 대역폭과 비교할 수 있는 지표 |
| 왼쪽 / 오른쪽 | out-of-place / in-place 결과 (보통 비슷함) |

- **큰 메시지(수백 MiB~GiB) 줄** 을 봅니다. 작은 메시지는 지연 시간 때문에 대역폭이 낮게 나오는 것이 정상입니다.
- 마지막의 `Avg bus bandwidth`는 모든 크기의 평균이라 참고용입니다.

#### ④ 무엇과 비교하는가

- **1노드:** 노드 안 NVLink만 사용합니다. GPU·드라이버·NCCL이 정상인지 확인하는 용도입니다.
- **2노드 이상:** EFA를 거칩니다. ②에서 EFA 경로인지 먼저 확인한 뒤 성능을 봅니다.
  - all-reduce는 NCCL이 NVLS tree(노드 안 합산은 NVLink Switch, 노드 간은 EFA)를 쓰면 busbw가 1노드와 비슷하게 나올 수 있습니다. 이때는 **algbw가 노드당 EFA 대역폭에 얼마나 가까운지** 로 판단합니다.
  - all-to-all은 노드 간 트래픽 비중이 커서 크게 낮게 나오는 것이 정상입니다. GPU당 노드 간 트래픽(2노드면 algbw의 절반)을 GPU당 EFA 몫(약 50 GB/s)과 비교합니다.
  - 기준값(4절)과 비교하고, 실 환경 측정값을 기준선으로 기록해 두십시오.

### 트러블슈팅

| 증상 | 원인 | 조치 |
|---|---|---|
| NCCL 로그에 `Using network Socket` | EFA 플러그인 미로드(TCP fallback) | 이미지의 NCCL 네트워크 플러그인이 EFA(ofi)용인지, IB용 `NCCL_*` 설정이 남아 있지 않은지 확인 |
| `NCCL_NET=IB` 등으로 통신 초기화 실패 | IB 경로 강제 | IB용 환경변수(`NCCL_NET=IB`, `NCCL_IB_*`) 제거. EFA 환경에서는 `NCCL_NET`을 지정하지 않는 것이 안전 |
| EFA는 타는데(`Libfabric`) 성능이 기준값의 1/4~1/10 | rank당 CPU 1코어 | `--cpus-per-task`를 vCPU ÷ GPU 수로 지정 |
| `slurmstepd: error: execve(): bash: No such file or directory` | `--export`에 `ALL` 없음 | `--export=ALL,...`로 수정 |
| 컨테이너 안 MPI 프로그램이 `opal_shmem_base_select failed` | 이미지의 Open MPI와 다른 Open MPI가 섞임 | 컨테이너 안 통신 확인은 MPI 없는 PyTorch 측정으로 수행 |
| `--mpi=pmix`가 동작하지 않음 | Slurm이 PMIx 지원으로 빌드되지 않음 | `srun --mpi=list`로 `pmix` 지원 확인. Slurm을 직접 빌드하면 `--with-pmix` 사용 |
| `/sys/class/infiniband/`나 컨테이너 `/dev/infiniband`에 EFA 외 다른 장치가 보임 | 일부 인스턴스에는 링크 DOWN 상태의 추가 RDMA 장치가 보임(EFA와 별개) | 조치 불필요. NCCL이 EFA를 선택하면 됨(`Using network Libfabric`, `found 8 nics`). IB 경로를 강제하지 말 것 |
| `#wrong`이 0이 아님 / `Out of bounds values`가 0이 아님 | 데이터 정확성 실패 | 성능과 무관하게 no-go. 하드웨어·드라이버·메모리 점검 |

### 체크리스트

- [ ] `srun --mpi=list`에 `pmix` 지원 확인
- [ ] 1노드 all-reduce 정상 (GPU·드라이버·NCCL 건전성)
- [ ] 호스트 2노드 all-reduce·all-to-all 측정 (`--cpus-per-task` 지정)
- [ ] NCCL 로그에 `Using network Libfabric`, `found 8 nics` 확인 (EFA 경로)
- [ ] `#wrong` / `Out of bounds values` 모두 0
- [ ] 컨테이너(학습 이미지) 2노드 측정이 호스트와 비슷함 (PyTorch 측정)
- [ ] `fi_pingpong` 노드 간 지연 측정
- [ ] 실 환경 측정값을 기준선으로 기록

> 📎 참고 공식 문서: [Get started with EFA and NCCL](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html) · [nccl-tests](https://github.com/NVIDIA/nccl-tests) · [`fi_pingpong(1)`](https://ofiwg.github.io/libfabric/main/man/fi_pingpong.1.html) · [NCCL 환경변수](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

## 체크리스트

- [ ] 노드 호스트 이름이 IPv4로 해석됨 — `NodeAddr`/`/etc/hosts` (1단계)
- [ ] 모든 노드가 동일한 `/etc/munge/munge.key`(소유자 munge, 권한 400), 공유 사본 삭제 (2단계)
- [ ] `slurm.conf` 노드 값이 `slurmd -C`와 일치, `gres.conf` GPU 등록 (3단계)
- [ ] `srun --mpi=list`에 `pmix` (4단계)
- [ ] `/etc/enroot/enroot.conf` 설정, 캐시·데이터는 로컬 NVMe, `/run/enroot` tmpfiles 등록 (5단계)
- [ ] Pyxis SPANK 플러그인 등록, `srun --help`에 `--container-image` (6단계)
- [ ] 2노드 컨테이너 실행에서 GPU 8개·`/dev/infiniband` 확인 (7단계)
- [ ] 잡 제출: `--cpus-per-task` 지정, `--export=ALL,...`, `--gres=gpu:8` (잡 제출)
- [ ] `NCCL_DEBUG=INFO` 로그에 `Using network Libfabric`·`found 8 nics` (잡 제출)
