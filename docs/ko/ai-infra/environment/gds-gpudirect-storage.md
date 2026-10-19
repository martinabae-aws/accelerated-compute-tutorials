---
title: GDS (GPUDirect Storage)
description: FSx for Lustre + GDS over EFA 점검 — nvidia-fs 드라이버·FSx 클라이언트 EFA 설정·cuFile을 호스트에서 먼저, 컨테이너에서 다음으로 검증하는 방법
---

# GDS (GPUDirect Storage)

GPUDirect Storage(GDS)는 스토리지에서 읽은 데이터를 **CPU 메모리를 거치지 않고 GPU 메모리로 직접 전송** 하는 기술입니다. 대규모 학습에서 체크포인트 저장·복원, 대용량 데이터셋 로딩 시 CPU 바운스 버퍼 복사를 없애 CPU 사용량을 줄이고, 여러 GPU가 동시에 I/O할 때 합산 대역폭을 끌어올립니다.

이 페이지는 Amazon FSx for Lustre(EFA 활성화 파일시스템)와 GDS 지원 GPU 인스턴스 조합에서 **GDS over EFA가 실제로 동작하는지** 를 점검하는 방법을 다룹니다. 원칙(왜 이렇게 점검하는가)을 본문에, 특정 시점의 버전·명령·기대 출력은 "검증 환경" 박스에 분리했습니다.

!!! note "검증 환경 (2026-10 기준)"
    아래 "실제 출력 예"는 특정 시점에 구축한 GPU 노드에서 실제로 얻은 정상 결과의 스냅샷입니다. 버전이 바뀌면 이 박스의 값만 갱신하면 되고, 본문의 점검 원칙은 그대로 유효합니다.

    - 대상: GDS 지원 GPU 인스턴스(검증 시 8-GPU, EFA 다중 인터페이스 노드) + Amazon FSx for Lustre (EFA 활성화 파일시스템)
    - 호스트 AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04). 이 AMI에는 Lustre 클라이언트, EFA 드라이버, GDS 드라이버(nvidia-fs)가 미리 설치되어 있습니다.
    - 기준 문서: AWS 「Configuring EFA clients」(FSx for Lustre 사용 설명서), NVIDIA GDS 문서
    - EFA 인터페이스 기준 개수: 8개 (인스턴스 유형에 따라 다름)
    - nvidia-fs 최소 버전: **2.24.2 이상** (FSx for Lustre 요구사항)

## 먼저 알아 둘 점

점검을 시작하기 전에 다음 전제를 이해하면 출력 해석이 쉬워집니다.

- **GPU 노드 1대의 호스트에서 먼저** 점검하십시오. 호스트에서 GDS가 동작하는 것을 확인한 다음, 컨테이너 안에서 확인합니다(마지막 섹션). 호스트가 깨져 있으면 컨테이너도 반드시 깨집니다.
- GDS over EFA가 동작하려면 다음 **세 요소** 가 모두 갖춰져야 합니다. 이 가이드도 이 순서로 점검합니다.
    1. **GDS 드라이버(nvidia-fs)**: 최소 버전 이상이 로드되어 있어야 합니다.
    2. **FSx 클라이언트의 EFA 설정**: `setup.sh --optimized-for-gds`를 실행해 Lustre 네트워크(LNet)에 EFA 인터페이스가 모두 잡혀 있어야 합니다.
    3. **cuFile(GDS 라이브러리)**: Lustre 경로를 GDS로 처리해야 합니다.
- GDS 지원 EFA 인스턴스에서 EFA 장치 이름은 `rdmap*`입니다. `mlx5_*`로 보이는 장치는 ConnectX-7(링크 DOWN)이므로 **GDS·EFA 점검과 무관** 합니다. 이 둘을 혼동하지 마십시오.
- **`gdscheck` 출력에는 `Unsupported`, `Disabled`, `false`가 많이 나옵니다.** 대부분 이 환경에서 쓰지 않는 다른 스토리지·NIC용 항목이라 정상입니다. 어떤 줄만 보면 되는지는 아래 "GDS 플랫폼 점검"에 정리했습니다.
- **처리량이 출력된다고 GDS가 동작한 것은 아닙니다.** cuFile은 GDS 경로를 쓸 수 없으면 CPU 경유(compat mode)로 넘어가도 정상적으로 처리량을 출력합니다. GDS 동작 여부는 "실제 GDS I/O 확인"의 **실행 전후 카운터 비교** 로 판단합니다.

### 점검 순서 요약

| 단계 | 확인 내용 | 정상 기준 |
|---|---|---|
| 1 | GDS 드라이버 | `nvidia_fs` 로드, 최소 버전 이상 |
| 2 | FSx 클라이언트 EFA 설정 | `lnetctl`에 `efa` 인터페이스가 기준 개수만큼, 서비스 active |
| 3 | GDS 플랫폼 점검 (`gdscheck -p`) | `DDN EXAScaler`에 `nvfs` 포함, 모든 GPU `supports GDS`, `Platform verification succeeded` |
| 4 | 실제 GDS I/O (`gdsio`) | GDS 실행 후 nvidia-fs의 `Mmap`·`Bar1-map`·`Sparse Reads`와 EFA 카운터가 증가 |
| 5 | 컨테이너 안 확인 | 컨테이너에서도 3·4단계와 같은 결과 |

## 1. GDS 드라이버(nvidia-fs) 확인

GDS는 커널 모듈 `nvidia-fs`가 GPU 드라이버와 연동되어야 동작합니다. 먼저 모듈 로드 여부와 버전을 확인합니다.

```bash
lsmod | grep nvidia_fs
grep "NVFS Driver" /proc/driver/nvidia-fs/stats
```

!!! note "실제 출력 예 — 정상 (2026-10 기준)"
    ```
    nvidia_fs             397312  0
    nvidia              15106048  1005 nvidia_uvm,efa_nv_peermem,nvidia_fs,gdrdrv,nvidia_modeset
    NVFS Driver(version: 2.30.2)
    ```

| 출력 | 의미 |
|---|---|
| `nvidia_fs  397312  0` | nvidia-fs 모듈이 로드되어 있습니다. 마지막 숫자 `0`은 지금 이 모듈을 쓰는 프로세스 수이며, I/O를 하지 않을 때 0인 것은 정상입니다 |
| `nvidia ... nvidia_fs,...` | GPU 드라이버(`nvidia`)를 nvidia-fs가 사용 중이라는 뜻입니다. `grep nvidia_fs`에 함께 걸려 나오는 줄이므로 신경 쓰지 않아도 됩니다 |
| `NVFS Driver(version: ...)` | nvidia-fs 버전. **최소 버전(검증 환경 박스 참고) 이상이면 정상** 입니다 |

`cat /proc/driver/nvidia-fs/version`도 버전을 보여 주지만 앞 두 자리만 나옵니다. 소스에서 빌드해 `insmod`로 올린 경우 `modinfo nvidia_fs`는 `module not found` 오류를 낼 수 있습니다. 모듈 파일이 시스템 모듈 경로에 설치되지 않았기 때문이며, 위 `lsmod`와 버전이 보이면 정상입니다.

### 모듈이 없거나 버전이 낮으면

먼저 로드를 시도합니다.

```bash
sudo modprobe nvidia_fs
lsmod | grep nvidia_fs
```

**로드되지 않거나 버전이 최소 버전 미만이면** 소스에서 빌드해 설치합니다(AWS 문서 절차).

```bash
git clone https://github.com/NVIDIA/gds-nvidia-fs.git
cd gds-nvidia-fs/src/
export NVFS_MAX_PEER_DEVS=128
export NVFS_MAX_PCI_DEPTH=16
sudo -E make
sudo insmod nvidia-fs.ko
```

**`insmod`로 올린 모듈은 재부팅하면 사라집니다.** 재부팅 후 `lsmod | grep nvidia_fs`를 다시 확인하고, 모든 GPU 노드에 같은 방식으로 적용하려면 AMI 또는 부팅 스크립트에 포함하십시오.

## 2. FSx 클라이언트 EFA 설정 확인

### 사전 조건

- FSx for Lustre 파일시스템이 **EFA 활성화로 생성** 되어 있어야 합니다(생성 후 변경 불가).
- 파일시스템과 GPU 노드의 보안 그룹이 **EFA 트래픽을 허용** 해야 합니다(FSx 사용 설명서 「EFA-enabled security groups」).

### 설정 상태 확인

Lustre 네트워크 계층(LNet)이 TCP가 아니라 EFA를 쓰도록 구성되어 있는지 확인합니다.

```bash
systemctl status configure-efa-fsx-lustre-client.service
sudo lnetctl net show --net efa
sudo lnetctl net show --net efa | grep -c "@efa"
df -hT <FSx 마운트 경로>
```

| 확인 | 정상 | 의미 |
|---|---|---|
| 서비스 상태 | `active` | 부팅 때 EFA 구성을 자동으로 하는 서비스가 동작 중 |
| `lnetctl net show --net efa` | `net type: efa` 아래에 `nid: <IP>@efa` 항목들이 있고 각 `status: up` | LNet이 EFA를 쓰도록 설정됨. 항목 하나가 EFA 인터페이스(`rdmap*`) 하나입니다 |
| `grep -c "@efa"` 결과 | 기준 개수(검증 환경 박스) | 인스턴스 기본값인 EFA 인터페이스가 모두 잡힘 |
| `df -hT` | `Type`이 `lustre`, `Filesystem`이 `<IP 또는 DNS>@tcp:/<마운트 이름>` | FSx for Lustre가 마운트됨. 마운트 주소에 `@tcp`가 보여도 정상입니다(마운트 주소 표기일 뿐, 실제 데이터는 위 EFA 인터페이스로 흐릅니다) |

!!! note "검증 환경 (2026-10 기준)"
    - `grep -c "@efa"` 결과 = **8** (EFA 인터페이스 8개가 모두 LNet에 잡힌 상태)
    - 뒤에서 쓰는 `lnetctl net show --net efa -v | grep send_count` 출력의 `send_count` 줄 개수도 EFA 인터페이스 개수와 같습니다(8줄이면 8개).

**`net type: efa`가 없고 `tcp`만 있거나, 서비스가 없으면** EFA 설정이 되어 있지 않은 상태입니다. 이 상태에서는 FSx 트래픽이 TCP로 흐르고 GDS over EFA도 동작하지 않습니다. 아래 "설정 방법"을 진행하십시오.

### 설정 방법 (GDS용)

```bash
curl -O https://docs.aws.amazon.com/fsx/latest/LustreGuide/samples/configure-efa-fsx-lustre-client.zip
unzip configure-efa-fsx-lustre-client.zip
cd configure-efa-fsx-lustre-client
sudo ./setup.sh --optimized-for-gds
```

이 스크립트는 다음을 수행합니다.

- Lustre 모듈 로드
- TCP와 EFA 인터페이스 구성 (인스턴스 기본 개수만큼)
- 재부팅 후에도 자동 구성되도록 systemd 서비스(`configure-efa-fsx-lustre-client.service`) 생성

GDS를 쓸 노드에는 옵션 없는 `sudo ./setup.sh`가 아니라 반드시 **`--optimized-for-gds`** 로 실행하십시오. 이 옵션이 GDS 경로에 필요한 EFA 구성을 적용합니다.

설정 후 위 "설정 상태 확인" 명령을 다시 실행합니다. 이미 마운트되어 있었다면 EFA 설정이 반영되도록 **언마운트 후 다시 마운트** 하십시오.

`/etc/fstab`으로 자동 마운트한다면 EFA 구성 서비스가 끝난 뒤 마운트되도록 옵션을 추가합니다.

```
<파일시스템 DNS 이름>@tcp:/<마운트 이름>  /fsx  lustre  defaults,relatime,flock,_netdev,x-systemd.automount,x-systemd.requires=configure-efa-fsx-lustre-client.service  0 0
```

### 문제가 있을 때 볼 로그

```bash
sudo journalctl -u configure-efa-fsx-lustre-client.service
sudo dmesg | grep -i -E "lnet|efa|lustre"
```

## 3. GDS 플랫폼 점검 (`gdscheck -p`)

`gdscheck`는 GDS를 쓰기 위한 플랫폼 조건(드라이버 버전, 스토리지 지원 여부, GPU, IOMMU 등)을 한 번에 점검합니다.

```bash
ls -d /usr/local/cuda*/gds/tools
/usr/local/cuda/gds/tools/gdscheck -p
```

`/usr/local/cuda/gds/tools`가 없으면 첫 줄에 나온 경로(예: `/usr/local/cuda-13.x/gds/tools`)를 쓰십시오.

### 한눈에 보기: 이 5줄만 확인하면 됩니다

출력이 길고 `Unsupported`·`Disabled`·`false`가 많지만, 실제로 봐야 하는 것은 다음 5줄입니다.

| # | 찾을 줄 | 정상 값 |
|---|---|---|
| ① | `nvidia_fs version` | 최소 버전 이상 |
| ② | `DDN EXAScaler` | **`nvfs`가 포함됨** (예: `nvfs, compat`) |
| ③ | `GPU index 0` ~ 마지막 GPU | 모든 GPU `supports GDS` |
| ④ | `IOMMU` | `disabled` |
| ⑤ | 마지막 줄 | `Platform verification succeeded` |

**이 5줄이 정상이면 나머지 줄에 `Unsupported`·`Disabled`·`false`가 있어도 GDS 플랫폼 점검은 통과입니다.**

### 구역별 해석

출력은 5개 구역으로 나뉩니다. 구역별로 무엇을 보는지 정리합니다.

**(가) 버전 정보**

| 줄 | 해석 |
|---|---|
| `GDS release version` | `gdscheck` 도구의 버전입니다. nvidia-fs stats 파일의 `GDS Version`과 다를 수 있으며, 도구가 들어 있는 CUDA 버전 차이일 뿐 **문제가 아닙니다** |
| `nvidia_fs version` | ① 확인. `nvidia_fs minimum version`은 이 도구가 요구하는 최소값이고, FSx for Lustre는 그보다 높은 **검증 환경 박스의 최소 버전** 이 필요합니다 |

**(나) DRIVER CONFIGURATION — 스토리지 종류별 지원 상태**

각 스토리지마다 지원 방식이 표시됩니다. 값의 뜻은 다음과 같습니다.

| 값 | 뜻 |
|---|---|
| `nvfs` | nvidia-fs를 통한 **GDS 경로를 쓸 수 있음** |
| `compat` | CPU 경유(호환 모드)로만 동작 |
| `Unsupported` / `Disabled` / `Not Loaded` / `Not configured` | 해당 기능을 쓰지 않음 |

이 환경에서 중요한 줄과 "정상이지만 무시해도 되는" 줄은 다음과 같습니다.

| 줄 | 해석 |
|---|---|
| **`DDN EXAScaler : nvfs, compat`** | **② 확인. 이 줄이 Lustre(FSx for Lustre)입니다.** gdscheck에는 "Lustre"라는 이름의 줄이 없고 "DDN EXAScaler"로 표시됩니다. `nvfs`가 있으므로 GDS 사용 가능. `compat`도 함께 있는 것은 "GDS가 안 되면 CPU 경유로 대신 처리할 수 있다"는 뜻이라 정상입니다 |
| NVMe, NVMeOF, SCSI, NFS, BeeGFS, WekaFS 등 `compat` | 이 노드에서 GDS 대상으로 쓰지 않는 스토리지입니다. **관계없음** |
| `Userspace RDMA : Unsupported` | cuFile이 사용자 공간에서 RDMA NIC를 직접 다루는 방식입니다(WekaFS·GPFS·NFS over RDMA용). **FSx for Lustre는 커널의 Lustre 클라이언트가 EFA로 RDMA를 처리하므로 이 방식을 쓰지 않습니다.** 정상 |
| `--Mellanox PeerDirect : Disabled` | Mellanox(InfiniBand/RoCE) NIC 전용 기능입니다. EFA를 쓰므로 정상 |
| `--rdma library : Not Loaded`, `--rdma devices : Not configured`, `--rdma_device_status : Up: 0 Down: 0` | 위 Userspace RDMA용 라이브러리·장치 설정입니다. 쓰지 않으므로 정상. **EFA가 없다는 뜻이 아닙니다.** EFA 경로는 2단계 `lnetctl`과 4단계 카운터로 확인합니다 |

**(다) CUFILE CONFIGURATION — `/etc/cufile.json` 설정값**

이 구역은 설정 파일 값을 그대로 보여 주는 것이며 **합격·불합격 판정 항목이 아닙니다.** 헷갈리기 쉬운 줄만 설명합니다.

| 줄 | 해석 |
|---|---|
| `use_pci_p2pdma : false`, `fs.lustre.use_pci_p2pdma : false` | nvidia-fs 없이 리눅스 커널 기본 P2P DMA를 쓰는 다른 방식입니다. 이 환경은 nvidia-fs 방식을 쓰므로 `false`가 정상 |
| `use_compat_mode : true` | GDS 경로가 실패하면 CPU 경유로 대신 처리합니다(기본값). 정상이지만, **이 때문에 처리량만으로는 GDS 동작을 판단할 수 없습니다**(4단계) |
| `force_compat_mode : false` | `true`이면 항상 CPU 경유. `false`가 정상 |
| `fs.weka...`, `fs.gpfs...`, `block.nvme...` 등 | 다른 스토리지용 설정. 관계없음 |

**(라) GPU INFO**

| 줄 | 해석 |
|---|---|
| `supports GDS` | ③ 확인. 모든 GPU에 있어야 합니다 |
| `bar size (MiB):...` | GPU 메모리를 PCIe로 노출하는 영역 크기입니다. 참고값 |
| `IOMMU State: Disabled` | 정상(GDS는 IOMMU가 꺼져 있을 때 동작) |

**(마) PLATFORM INFO**

| 줄 | 해석 |
|---|---|
| `IOMMU: disabled` | ④ 확인 |
| `Nvidia Driver Info Status: Supported` | GPU 드라이버가 GDS를 지원함 |
| `Cuda Driver Version Installed: ...` | 호스트 드라이버의 CUDA 버전 |
| `Platform verification succeeded` | ⑤ 확인. 플랫폼 점검 전체 통과 |

!!! note "실제 출력 예 — 정상 (2026-10 기준)"
    ```
     GDS release version: 1.17.0.44
     nvidia_fs version:  2.30 nvidia_fs minimum version: 2.12
     Platform: x86_64
     ...
     DDN EXAScaler      : nvfs, compat
     Userspace RDMA     : Unsupported
     --Mellanox PeerDirect : Disabled
     --rdma library        : Not Loaded (libcufile_rdma.so)
     --rdma devices        : Not configured
     --rdma_device_status  : Up: 0 Down: 0
     ...
     GPU index 0 NVIDIA <GPU> bar:1 bar size (MiB):262144 supports GDS, IOMMU State: Disabled
     ...
     GPU index 7 NVIDIA <GPU> bar:1 bar size (MiB):262144 supports GDS, IOMMU State: Disabled
     ...
     IOMMU: disabled
     Nvidia Driver Info Status: Supported(Nvidia Open Driver Installed)
     Cuda Driver Version Installed:  13020
     Platform: <GDS 지원 GPU 인스턴스>, Arch: x86_64(Linux 7.0.0-1013-aws)
     Platform verification succeeded
    ```

### 비정상일 때

| 증상 | 확인할 것 |
|---|---|
| `DDN EXAScaler`가 `compat`만 있음 (`nvfs` 없음) | 1단계 nvidia-fs 로드 여부 / Lustre 마운트 여부(`df -hT`) / 2단계 `--optimized-for-gds` 설정 여부 |
| `supports GDS`가 없는 GPU가 있음 | `nvidia-smi`로 해당 GPU 상태 확인 후 AWS에 공유 |
| `Platform verification failed` | 출력 전체를 AWS에 공유 |

## 4. 실제 GDS I/O 확인 (`gdsio`)

GDS가 실제로 동작하는지는 **gdsio를 GDS 모드로 실행하기 전과 후에 카운터 두 가지를 비교** 해서 판단합니다. 처리량 출력만으로는 compat mode(CPU 경유)와 구분할 수 없기 때문입니다.

- **nvidia-fs 카운터** (`/proc/driver/nvidia-fs/stats`): GPU 메모리로 직접 I/O(GDS 경로)를 했는지
- **EFA 카운터** (`lnetctl ... send_count / recv_count`): FSx 트래픽이 EFA로 흘렀는지

### gdsio 주요 옵션

| 옵션 | 의미 |
|---|---|
| `-f <파일>` | 테스트 파일 경로 (FSx 마운트 아래) |
| `-d <번호>` | GPU 번호 (한 번에 하나) |
| `-w <수>` | 워커 스레드 수 |
| `-s <크기>` | 파일 크기 (예: `10G`) |
| `-i <크기>` | I/O 한 번의 크기 (예: `1M`) |
| `-x <번호>` | **전송 방식.** `0` = GDS(GPU 직접), `1` = CPU 경유 |
| `-I <번호>` | **I/O 종류.** `0` = 읽기, `1` = 쓰기 |

### 실행 순서

`/fsx`는 실제 FSx 마운트 경로로 바꾸십시오. 쓰기가 테스트 파일을 만들기 때문에 쓰기를 먼저 실행합니다.

```bash
cat /proc/driver/nvidia-fs/stats
sudo lnetctl net show --net efa -v | grep -E "send_count|recv_count"

/usr/local/cuda/gds/tools/gdsio -f /fsx/gds_test -d 0 -w 4 -s 10G -i 1M -x 0 -I 1
/usr/local/cuda/gds/tools/gdsio -f /fsx/gds_test -d 0 -w 4 -s 10G -i 1M -x 0 -I 0

cat /proc/driver/nvidia-fs/stats
sudo lnetctl net show --net efa -v | grep -E "send_count|recv_count"

/usr/local/cuda/gds/tools/gdsio -f /fsx/gds_test -d 0 -w 4 -s 10G -i 1M -x 1 -I 0

cat /proc/driver/nvidia-fs/stats
```

1. 실행 전 카운터 기록
2. GDS 쓰기 → GDS 읽기
3. 실행 후 카운터 기록 (1과 비교)
4. 비교용 CPU 경유 읽기
5. CPU 실행 후 카운터 기록 (3과 비교 → GDS 카운터는 그대로여야 함)

### gdsio 출력 읽는 법

| 필드 | 의미 |
|---|---|
| `IoType` | `WRITE` 쓰기 / `READ` 읽기 (`-I` 값) |
| `XferType` | `GPUD` = GDS 모드로 **요청** 함(`-x 0`), `CPUONLY` = CPU 경유(`-x 1`). **`GPUD`로 나와도 compat mode로 대신 처리됐을 수 있으므로 이것만으로 GDS 동작을 판단하지 않습니다** |
| `Threads` | 워커 스레드 수 (`-w`) |
| `DataSetSize` | 실제 처리한 양 / 지정한 양(KiB). 거의 같으면 정상 |
| `IOSize` | I/O 한 번의 크기 (`-i`) |
| `Throughput` | 이 gdsio 프로세스 하나의 처리량(GiB/s) |
| `Avg_Latency` | I/O 한 번의 평균 지연(µs) |
| `ops` | 처리한 I/O 횟수 |
| `total_time` | 걸린 시간(초) |

!!! note "실제 출력 예 — 정상 (2026-10 기준)"
    ```
    IoType: WRITE XferType: GPUD Threads: 4 DataSetSize: 10285056/10485760(KiB) IOSize: 1024(KiB) Throughput: 3.202468 GiB/sec, Avg_Latency: 1218.884017 usecs ops: 10044 total_time 3.062823 secs
    IoType: READ XferType: GPUD Threads: 4 DataSetSize: 10383360/10485760(KiB) IOSize: 1024(KiB) Throughput: 3.189802 GiB/sec, Avg_Latency: 1224.218035 usecs ops: 10140 total_time 3.104376 secs
    IoType: READ XferType: CPUONLY Threads: 4 DataSetSize: 10434560/10485760(KiB) IOSize: 1024(KiB) Throughput: 3.706651 GiB/sec, Avg_Latency: 1052.802307 usecs ops: 10190 total_time 2.684680 secs
    ```

**GDS 처리량이 CPU보다 높지 않아도 GDS 동작 실패가 아닙니다.** GPU 1개·파일 1개 테스트에서는 FSx 파일시스템 쪽(파일이 저장된 OST 수, 파일시스템 처리량 한도)이 병목이 되는 경우가 많습니다. GDS의 이점은 CPU 사용량 감소와, 여러 GPU가 동시에 I/O할 때의 합산 대역폭에서 나타납니다(아래 "GPU 다중 동시 테스트").

### nvidia-fs 카운터 읽는 법

실행 전후 `/proc/driver/nvidia-fs/stats`를 비교합니다. 다음 세 값이 GDS 경로 사용의 핵심 증거입니다.

| 줄 | 의미 | 판단 |
|---|---|---|
| `Mmap` n / `Bar1-map` n | GPU 메모리 버퍼를 nvidia-fs에 등록한 횟수 | GDS 실행(쓰기·읽기 공통) 후 증가해야 함. CPU 실행 후에는 늘지 않음 — **GDS 경로 사용의 핵심 증거** |
| `Sparse Reads` n | nvidia-fs가 처리한 읽기 I/O 수 | GDS 읽기 후 증가해야 함(예: 10 GiB ÷ 1 MiB = 10240). `holes=0`은 정상 |
| `err=`, `Error` 줄 | 오류 횟수 | 모두 0이어야 함 |
| `Reads` / `Writes` | 읽기·쓰기 오류 수 | `IO stats: Disabled`(기본값)에서는 오류 수만 나오고 횟수는 나오지 않음. 0이면 정상 |
| `IO stats: Disabled` | 상세 읽기·쓰기 통계 꺼짐 | 기본값이며 정상. 위 세 값으로 판단 가능 |
| `Mellanox PeerDirect Supported: False` | Mellanox NIC 전용 기능 | EFA 환경에서는 정상 |
| `Active Process`, `Ops`, `Registered_MiB`, `Cache_MiB` | **지금 이 순간** 실행 중인 GDS 작업 | gdsio가 끝난 뒤에는 0이 정상. 누적값이 아니므로 판단에 쓰지 않음 |

!!! note "실제 출력 예 — 정상 (2026-10 기준)"
    **실행 전**
    ```
    GDS Version: 1.19.1.25
    NVFS statistics(ver: 4.0)
    NVFS Driver(version: 2.30.2)
    Mellanox PeerDirect Supported: False
    IO stats: Disabled, peer IO stats: Disabled
    ...
    Sparse Reads  : n=0 io=0 holes=0 pages=0
    Mmap          : n=4 ok=4 err=0 munmap=4
    Bar1-map      : n=4 ok=4 err=0 free=4 callbacks=0 active=0 delay-frees=0
    Ops           : Read=0 Write=0 BatchIO=0
    ```

    **GDS 쓰기·읽기(10 GiB씩) 실행 후 (바뀐 줄만)**
    ```
    Sparse Reads  : n=10240 io=0 holes=0 pages=0
    Mmap          : n=12 ok=12 err=0 munmap=12
    Bar1-map      : n=12 ok=12 err=0 free=12 callbacks=0 active=0 delay-frees=0
    ```
    - `Mmap`·`Bar1-map`: 4 → 12 (+8). GDS 실행 2회 × 스레드 4개 = 8. GDS 경로 사용의 증거.
    - `Sparse Reads`: 0 → 10240 (10 GiB ÷ 1 MiB). GDS 읽기의 증거.

    **CPU 경유(`-x 1`) 실행 후:** 위 값에서 **변화 없음** (CPU 경유는 nvidia-fs를 거치지 않음).

**판단:** GDS 모드 실행 후 `Mmap`·`Bar1-map`이 늘고, 읽기 후 `Sparse Reads`가 늘었으며, CPU 실행 후에는 그대로이면 **GDS 경로가 동작한 것** 입니다. 쓰기 횟수까지 직접 보려면 `echo 1 | sudo tee /sys/module/nvidia_fs/parameters/rw_stats_enabled`로 상세 통계를 켠 뒤(재부팅하면 꺼짐) 다시 실행하십시오. `IO stats: Enabled`로 바뀌고 `Reads`·`Writes` 줄에 횟수가 추가로 나옵니다.

### EFA 카운터 읽는 법

`lnetctl net show --net efa -v`의 `send_count`/`recv_count`는 각 EFA 인터페이스를 통해 오간 LNet 메시지 수입니다(바이트가 아님). 실행 전후를 비교해 FSx 트래픽이 EFA로 흘렀는지 확인합니다.

| 항목 | 해석 |
|---|---|
| `send_count`/`recv_count` 줄 쌍의 개수 | EFA 인터페이스 개수. 기준 개수면 모두 LNet에 잡혀 있음 |
| 크게 늘어난 인터페이스가 있음 | **FSx 트래픽이 EFA로 흘렀다는 증거** |
| 전체 중 일부만 크게 늘어남 | **정상.** GPU 1개(`-d 0`)로 테스트했기 때문에 그 GPU와 PCIe로 가까운 EFA 인터페이스가 쓰였습니다. GPU를 모두 동시에 쓰면 모든 인터페이스가 늘어납니다(아래) |
| 나머지가 조금씩 늘어남 | 연결 유지용 메시지. 정상 |
| 모든 인터페이스가 거의 그대로 | FSx 트래픽이 EFA로 흐르지 않음 → 2단계 점검 |

!!! note "실제 출력 예 — 정상 (2026-10 기준, EFA 8개 노드)"
    GPU 1개로 GDS 쓰기·읽기 후, 8개 중 그 GPU에 가까운 2개 인터페이스만 크게 증가:

    | EFA 인터페이스 | send_count 전 → 후 | recv_count 전 → 후 |
    |---|---|---|
    | 1 | 53 → 56 | 53 → 56 |
    | 2 | 53 → 56 | 53 → 56 |
    | 3 | 53 → 55 | 53 → 55 |
    | 4 | 52 → 55 | 52 → 55 |
    | **5** | **2,063 → 17,442** | **4,111 → 24,611** |
    | **6** | **2,062 → 17,407** | **4,110 → 24,574** |
    | 7 | 14 → 20 | 14 → 20 |
    | 8 | 14 → 19 | 14 → 19 |

### (선택) GPU 다중 동시 테스트 — 노드 합산 처리량과 EFA 전체 사용 확인

`-d`에는 GPU 하나만 지정할 수 있으므로, gdsio를 GPU 개수만큼 동시에 실행합니다. 모든 EFA 인터페이스가 고르게 쓰이고 합산 처리량이 올라가는 것을 확인합니다. `/fsx`는 실제 마운트 경로로 바꾸십시오.

```bash
mkdir -p /fsx/gds8
lfs setstripe -c -1 /fsx/gds8
sudo lnetctl net show --net efa -v | grep -E "send_count|recv_count" > /tmp/efa_before.txt

for i in 0 1 2 3 4 5 6 7; do
  /usr/local/cuda/gds/tools/gdsio -f /fsx/gds8/gds_test_$i -d $i -w 4 -s 10G -i 1M -x 0 -I 1 > /tmp/gds_w_$i.log 2>&1 &
done
wait
cat /tmp/gds_w_*.log
awk -F'Throughput: ' '/Throughput/{split($2,a," "); s+=a[1]} END{print "WRITE total GiB/s:", s}' /tmp/gds_w_*.log

for i in 0 1 2 3 4 5 6 7; do
  /usr/local/cuda/gds/tools/gdsio -f /fsx/gds8/gds_test_$i -d $i -w 4 -s 10G -i 1M -x 0 -I 0 > /tmp/gds_r_$i.log 2>&1 &
done
wait
cat /tmp/gds_r_*.log
awk -F'Throughput: ' '/Throughput/{split($2,a," "); s+=a[1]} END{print "READ total GiB/s:", s}' /tmp/gds_r_*.log

sudo lnetctl net show --net efa -v | grep -E "send_count|recv_count" > /tmp/efa_after.txt
paste /tmp/efa_before.txt /tmp/efa_after.txt
```

> 위 루프의 GPU 번호 범위(`0 1 2 ... 7`)는 노드의 GPU 개수에 맞춰 조정하십시오.

| 명령·출력 | 해석 |
|---|---|
| `lfs setstripe -c -1 /fsx/gds8` | 이 디렉터리에 새로 만드는 파일을 모든 OST에 나눠 저장합니다. 파일 하나가 OST 하나에 몰려 병목이 되는 것을 막습니다. 다른 경로에는 영향 없음 |
| `cat /tmp/gds_*_*.log` | GPU별 결과. 모두 `XferType: GPUD`, 오류 없음이어야 합니다 |
| `WRITE total GiB/s:` / `READ total GiB/s:` | **노드 1대의 합산 처리량.** gdsio에는 합산값이 없어 각 `Throughput`을 더한 값입니다. 상한은 FSx 파일시스템의 처리량 한도(용량 × TiB당 처리량)입니다 |
| `paste` 결과 | 왼쪽이 실행 전, 오른쪽이 실행 후. **모든 인터페이스가 크게 늘어야** GPU별로 EFA가 고르게 쓰인 것입니다 |

정리:

```bash
rm -rf /fsx/gds8
```

### GDS 경로가 안 탈 때: compat mode를 끄고 원인 확인

GDS 실행 후에도 `Mmap`·`Bar1-map`·`Sparse Reads`가 그대로이면, gdsio가 처리량을 출력했더라도 CPU 경유(compat mode)로 실행된 것입니다. compat mode를 끄면 GDS 경로가 실패할 때 CPU로 넘어가지 않고 에러를 내므로 원인을 볼 수 있습니다.

```bash
cp /etc/cufile.json /tmp/cufile.json
```

`/tmp/cufile.json`의 `"properties"` 안에서 다음 값을 바꿉니다. 같은 파일의 `"logging"` 안 `"level"`을 `"TRACE"`로 바꾸면 자세한 로그가 남습니다.

```json
"allow_compat_mode": false,
```

```bash
cd /tmp
CUFILE_ENV_PATH_JSON=/tmp/cufile.json /usr/local/cuda/gds/tools/gdsio -f /fsx/gds_test -d 0 -w 4 -s 1G -i 1M -x 0 -I 0
ls /tmp/cufile*.log
```

`CUFILE_ENV_PATH_JSON`으로 복사본을 지정하므로 `/etc/cufile.json` 원본은 바뀌지 않습니다. 에러 메시지와 `cufile.log`를 AWS에 공유하면 원인 확인에 도움이 됩니다.

### 테스트 파일 정리

```bash
rm -f /fsx/gds_test
```

## 5. 컨테이너(Pyxis/Enroot) 안에서 확인

호스트에서 앞 단계가 모두 정상인 다음 진행합니다. 사용자 학습 코드는 보통 컨테이너 안에서 돌기 때문에, 호스트의 nvidia-fs 장치와 GDS 도구가 컨테이너로 전달되는지 확인합니다.

```bash
srun -N1 --gres=gpu:8 --container-image=<학습 이미지> --container-mounts=/fsx:/fsx \
  bash -c 'ls -l /dev/nvidia-fs* | head -3; ls -d /usr/local/cuda*/gds/tools; grep "NVFS Driver" /proc/driver/nvidia-fs/stats'
```

> `--gres=gpu:` 값은 노드의 GPU 개수에 맞추십시오.

| 출력 | 정상 | 의미 |
|---|---|---|
| `ls -l /dev/nvidia-fs*` | 장치 파일 목록이 나옴 | 호스트의 nvidia-fs 장치가 컨테이너로 전달됨 |
| `ls -d .../gds/tools` | 경로가 나옴 | 컨테이너 안에 GDS 도구(`gdscheck`, `gdsio`)가 있음 |
| `NVFS Driver(version: ...)` | 호스트와 같은 버전 | 호스트 드라이버를 보고 있음 |

이 조건이 맞으면 컨테이너 안에서 3단계(`gdscheck -p`)와 4단계(`gdsio`)를 같은 방법으로 실행합니다. 카운터(`/proc/driver/nvidia-fs/stats`, `lnetctl`)는 **호스트에서** 확인하십시오.

**`/dev/nvidia-fs*`가 보이지 않으면** Enroot 설정(`ENROOT_RESTRICT_DEV` 등)에서 장치가 컨테이너로 전달되는지 확인하십시오.

## 6. 결과 판정 요약

| 확인 | 정상 | 비정상일 때 의미 | 조치 |
|---|---|---|---|
| `lsmod \| grep nvidia_fs` | 모듈 보임 | GDS 드라이버 없음 | 1단계 |
| `NVFS Driver(version)` | 최소 버전 이상 | 버전 낮음 | 1단계 소스 빌드 |
| `lnetctl ... grep -c "@efa"` | 기준 개수 | FSx가 EFA를 안 쓰거나 일부만 씀 | 2단계 `setup.sh --optimized-for-gds` |
| `gdscheck -p` `DDN EXAScaler` | `nvfs` 포함 | Lustre에 GDS 경로 없음 | 3단계 "비정상일 때" |
| `gdscheck -p` 모든 GPU | `supports GDS` | 일부 GPU 문제 | AWS 공유 |
| `gdscheck -p` 마지막 줄 | `Platform verification succeeded` | 플랫폼 점검 실패 | AWS 공유 |
| GDS 실행 후 `Mmap`·`Bar1-map` | 증가 | compat mode(CPU 경유)로 실행됨 | "compat mode를 끄고 원인 확인" |
| GDS 읽기 후 `Sparse Reads` | 증가 | 읽기가 GDS 경로를 안 탐 | "compat mode를 끄고 원인 확인" |
| GDS 실행 후 EFA `send_count`/`recv_count` | 일부 이상 크게 증가 (전 GPU 동시면 전체 증가) | FSx 트래픽이 EFA로 안 흐름 | 2단계 |
| `Unsupported` / `Disabled` / `false` 항목 | — | **3단계에 정리한 항목이면 정상** | 없음 |

## 7. AWS에 공유해 주실 정보

문제가 해결되지 않으면 다음 출력을 함께 보내 주십시오.

1. `grep "NVFS Driver" /proc/driver/nvidia-fs/stats`
2. `sudo lnetctl net show`
3. `systemctl status configure-efa-fsx-lustre-client.service`
4. `gdscheck -p` 전체 출력
5. gdsio 실행 결과와, 실행 전후 `/proc/driver/nvidia-fs/stats` 및 `lnetctl ... send_count|recv_count` 값
6. (compat mode를 끄고 재현했다면) 에러 메시지와 `cufile.log`

## 참고 문서

| 문서 | URL |
|---|---|
| FSx for Lustre — Configuring EFA clients | https://docs.aws.amazon.com/fsx/latest/LustreGuide/configure-efa-clients.html |
| FSx for Lustre — EFA 파일시스템 | https://docs.aws.amazon.com/fsx/latest/LustreGuide/efa-file-systems.html |
| NVIDIA GDS 트러블슈팅 가이드 | https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/ |
| NVIDIA GDS 설정 가이드 (cufile.json) | https://docs.nvidia.com/gpudirect-storage/configuration-guide/ |
