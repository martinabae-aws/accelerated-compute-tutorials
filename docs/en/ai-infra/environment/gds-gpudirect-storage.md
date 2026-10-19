---
title: GDS (GPUDirect Storage)
description: Checking FSx for Lustre + GDS over EFA — how to verify the nvidia-fs driver, FSx client EFA configuration, and cuFile first on the host and then inside the container
---

# GDS (GPUDirect Storage)

GPUDirect Storage (GDS) is a technology that transfers data read from storage **directly into GPU memory without going through CPU memory**. In large-scale training, it eliminates the CPU bounce-buffer copy when saving and restoring checkpoints or loading large datasets, which lowers CPU usage and raises aggregate bandwidth when multiple GPUs do I/O at the same time.

This page covers how to check **whether GDS over EFA actually works** on a combination of Amazon FSx for Lustre (an EFA-enabled file system) and a GDS-capable GPU instance. The rationale (why we check it this way) is in the body, while the versions, commands, and expected output at a specific point in time are separated out into "validated environment" boxes.

!!! note "Validated environment (as of 2026-10)"
    The "actual output examples" below are snapshots of normal results obtained on a GPU node built at a specific point in time. When versions change, you only need to update the values in this box; the inspection rationale in the body remains valid.

    - Target: GDS-capable GPU instance (8-GPU, EFA multi-interface node at the time of validation) + Amazon FSx for Lustre (an EFA-enabled file system)
    - Host AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04). This AMI comes with the Lustre client, EFA driver, and GDS driver (nvidia-fs) preinstalled.
    - Reference docs: AWS "Configuring EFA clients" (FSx for Lustre User Guide), NVIDIA GDS documentation
    - Baseline EFA interface count: 8 (varies by instance type)
    - Minimum nvidia-fs version: **2.24.2 or later** (FSx for Lustre requirement)

## Know these first

Understanding the following assumptions before you start will make the output easier to interpret.

- **Check on the host of a single GPU node first.** After confirming GDS works on the host, check inside the container (last section). If the host is broken, the container will be broken too.
- For GDS over EFA to work, all **three elements** below must be in place. This guide also inspects them in this order.
    1. **GDS driver (nvidia-fs)**: the minimum version or later must be loaded.
    2. **EFA configuration of the FSx client**: run `setup.sh --optimized-for-gds` so that all EFA interfaces are attached to the Lustre network (LNet).
    3. **cuFile (GDS library)**: it must handle the Lustre path over GDS.
- On a GDS-capable EFA instance, the EFA device name is `rdmap*`. Devices shown as `mlx5_*` are ConnectX-7 (link DOWN) and are therefore **unrelated to GDS/EFA inspection**. Do not confuse the two.
- **The `gdscheck` output contains many `Unsupported`, `Disabled`, and `false` entries.** Most are normal — they are entries for other storage/NIC types not used in this environment. Which lines you actually need to look at is summarized in "GDS platform inspection" below.
- **Throughput being reported does not mean GDS is working.** When cuFile cannot use the GDS path, it falls back to going through the CPU (compat mode) and still reports throughput normally. Whether GDS is actually working is determined by the **before/after counter comparison** in "Verifying actual GDS I/O".

### Inspection order summary

| Step | What to check | Normal criteria |
|---|---|---|
| 1 | GDS driver | `nvidia_fs` loaded, minimum version or later |
| 2 | FSx client EFA config | `lnetctl` has as many `efa` interfaces as the baseline count, service active |
| 3 | GDS platform inspection (`gdscheck -p`) | `DDN EXAScaler` includes `nvfs`, every GPU `supports GDS`, `Platform verification succeeded` |
| 4 | Actual GDS I/O (`gdsio`) | After running GDS, nvidia-fs's `Mmap`/`Bar1-map`/`Sparse Reads` and the EFA counters increase |
| 5 | Check inside container | Same results as steps 3 and 4 inside the container |

## 1. Verify the GDS driver (nvidia-fs)

GDS works only when the kernel module `nvidia-fs` is linked with the GPU driver. First check whether the module is loaded and its version.

```bash
lsmod | grep nvidia_fs
grep "NVFS Driver" /proc/driver/nvidia-fs/stats
```

!!! note "Actual output example — normal (as of 2026-10)"
    ```
    nvidia_fs             397312  0
    nvidia              15106048  1005 nvidia_uvm,efa_nv_peermem,nvidia_fs,gdrdrv,nvidia_modeset
    NVFS Driver(version: 2.30.2)
    ```

| Output | Meaning |
|---|---|
| `nvidia_fs  397312  0` | The nvidia-fs module is loaded. The trailing `0` is the number of processes currently using this module; it being 0 when no I/O is in progress is normal |
| `nvidia ... nvidia_fs,...` | Means nvidia-fs is in use by the GPU driver (`nvidia`). This line is caught together by `grep nvidia_fs`, so you can ignore it |
| `NVFS Driver(version: ...)` | The nvidia-fs version. **It is normal if it is at or above the minimum version (see the validated environment box)** |

`cat /proc/driver/nvidia-fs/version` also shows the version, but only the first two digits. If you built from source and loaded it with `insmod`, `modinfo nvidia_fs` may return a `module not found` error. That is because the module file was not installed into the system module path; if the `lsmod` output and the version above are visible, it is normal.

### If the module is missing or the version is too low

First try to load it.

```bash
sudo modprobe nvidia_fs
lsmod | grep nvidia_fs
```

**If it is not loaded or the version is below the minimum**, build and install from source (AWS documented procedure).

```bash
git clone https://github.com/NVIDIA/gds-nvidia-fs.git
cd gds-nvidia-fs/src/
export NVFS_MAX_PEER_DEVS=128
export NVFS_MAX_PCI_DEPTH=16
sudo -E make
sudo insmod nvidia-fs.ko
```

**A module loaded with `insmod` disappears on reboot.** After reboot, check `lsmod | grep nvidia_fs` again, and to apply it the same way to all GPU nodes, include it in the AMI or the boot script.

## 2. Verify the FSx client EFA configuration

### Prerequisites

- The FSx for Lustre file system must have been **created with EFA enabled** (it cannot be changed after creation).
- The security groups of the file system and the GPU node must **allow EFA traffic** (FSx User Guide "EFA-enabled security groups").

### Check the configuration state

Verify that the Lustre network layer (LNet) is configured to use EFA rather than TCP.

```bash
systemctl status configure-efa-fsx-lustre-client.service
sudo lnetctl net show --net efa
sudo lnetctl net show --net efa | grep -c "@efa"
df -hT <FSx mount path>
```

| Check | Normal | Meaning |
|---|---|---|
| Service status | `active` | The service that automatically configures EFA at boot is running |
| `lnetctl net show --net efa` | Under `net type: efa` there are `nid: <IP>@efa` entries, each with `status: up` | LNet is configured to use EFA. Each entry is one EFA interface (`rdmap*`) |
| `grep -c "@efa"` result | Baseline count (validated environment box) | All of the instance's default EFA interfaces are attached |
| `df -hT` | `Type` is `lustre`, `Filesystem` is `<IP or DNS>@tcp:/<mount name>` | FSx for Lustre is mounted. Seeing `@tcp` in the mount address is still normal (it is only the mount address notation; the actual data flows over the EFA interfaces above) |

!!! note "Validated environment (as of 2026-10)"
    - `grep -c "@efa"` result = **8** (all 8 EFA interfaces attached to LNet)
    - The number of `send_count` lines in the `lnetctl net show --net efa -v | grep send_count` output used later also equals the number of EFA interfaces (8 lines means 8 interfaces).

**If there is no `net type: efa` and only `tcp`, or the service is missing**, EFA is not configured. In this state FSx traffic flows over TCP and GDS over EFA will not work either. Proceed with "How to configure" below.

### How to configure (for GDS)

```bash
curl -O https://docs.aws.amazon.com/fsx/latest/LustreGuide/samples/configure-efa-fsx-lustre-client.zip
unzip configure-efa-fsx-lustre-client.zip
cd configure-efa-fsx-lustre-client
sudo ./setup.sh --optimized-for-gds
```

This script does the following.

- Loads the Lustre modules
- Configures the TCP and EFA interfaces (as many as the instance default count)
- Creates a systemd service (`configure-efa-fsx-lustre-client.service`) so that it is auto-configured even after reboot

On nodes that will use GDS, be sure to run **`--optimized-for-gds`**, not the plain `sudo ./setup.sh` without options. This option applies the EFA configuration required for the GDS path.

After configuring, run the "Check the configuration state" commands again. If it was already mounted, **unmount and remount** so that the EFA configuration takes effect.

If you auto-mount via `/etc/fstab`, add options so that it is mounted after the EFA configuration service finishes.

```
<file system DNS name>@tcp:/<mount name>  /fsx  lustre  defaults,relatime,flock,_netdev,x-systemd.automount,x-systemd.requires=configure-efa-fsx-lustre-client.service  0 0
```

### Logs to check when there is a problem

```bash
sudo journalctl -u configure-efa-fsx-lustre-client.service
sudo dmesg | grep -i -E "lnet|efa|lustre"
```

## 3. GDS platform inspection (`gdscheck -p`)

`gdscheck` inspects the platform conditions for using GDS (driver version, storage support, GPU, IOMMU, etc.) all at once.

```bash
ls -d /usr/local/cuda*/gds/tools
/usr/local/cuda/gds/tools/gdscheck -p
```

If `/usr/local/cuda/gds/tools` does not exist, use the path shown in the first line (e.g. `/usr/local/cuda-13.x/gds/tools`).

### At a glance: you only need to check these 5 lines

The output is long and contains many `Unsupported`, `Disabled`, and `false` entries, but the lines you actually need to look at are the following 5.

| # | Line to find | Normal value |
|---|---|---|
| ① | `nvidia_fs version` | Minimum version or later |
| ② | `DDN EXAScaler` | **includes `nvfs`** (e.g. `nvfs, compat`) |
| ③ | `GPU index 0` ~ last GPU | Every GPU `supports GDS` |
| ④ | `IOMMU` | `disabled` |
| ⑤ | Last line | `Platform verification succeeded` |

**If these 5 lines are normal, the GDS platform inspection passes even if the remaining lines contain `Unsupported`, `Disabled`, or `false`.**

### Section-by-section interpretation

The output is divided into 5 sections. Here is what to look at in each.

**(a) Version information**

| Line | Interpretation |
|---|---|
| `GDS release version` | The version of the `gdscheck` tool. It may differ from the `GDS Version` in the nvidia-fs stats file; this is just a difference in the CUDA version the tool is in and is **not a problem** |
| `nvidia_fs version` | Check ①. `nvidia_fs minimum version` is the minimum this tool requires, and FSx for Lustre needs the higher **minimum version of the validated environment box** |

**(b) DRIVER CONFIGURATION — support status by storage type**

The support method is shown for each storage. The meanings of the values are as follows.

| Value | Meaning |
|---|---|
| `nvfs` | **The GDS path via nvidia-fs can be used** |
| `compat` | Works only via the CPU (compatibility mode) |
| `Unsupported` / `Disabled` / `Not Loaded` / `Not configured` | The feature is not used |

The lines that matter in this environment and the lines that are "normal but can be ignored" are as follows.

| Line | Interpretation |
|---|---|
| **`DDN EXAScaler : nvfs, compat`** | **Check ②. This line is Lustre (FSx for Lustre).** gdscheck has no line literally named "Lustre"; it is shown as "DDN EXAScaler". Since `nvfs` is present, GDS can be used. `compat` also being present means "if GDS does not work, it can process via the CPU instead" and is normal |
| NVMe, NVMeOF, SCSI, NFS, BeeGFS, WekaFS, etc. `compat` | Storage not used as a GDS target on this node. **Irrelevant** |
| `Userspace RDMA : Unsupported` | A way for cuFile to drive the RDMA NIC directly from user space (for WekaFS/GPFS/NFS over RDMA). **FSx for Lustre does not use this method because the kernel Lustre client handles RDMA over EFA.** Normal |
| `--Mellanox PeerDirect : Disabled` | A feature only for Mellanox (InfiniBand/RoCE) NICs. Normal since EFA is used |
| `--rdma library : Not Loaded`, `--rdma devices : Not configured`, `--rdma_device_status : Up: 0 Down: 0` | The library/device settings for the Userspace RDMA above. Normal since they are not used. **This does not mean EFA is absent.** The EFA path is confirmed by `lnetctl` in step 2 and by the counters in step 4 |

**(c) CUFILE CONFIGURATION — `/etc/cufile.json` settings**

This section simply shows the settings file values and **is not a pass/fail judgment section.** Only the easily confused lines are explained.

| Line | Interpretation |
|---|---|
| `use_pci_p2pdma : false`, `fs.lustre.use_pci_p2pdma : false` | A different method that uses the Linux kernel's built-in P2P DMA without nvidia-fs. This environment uses the nvidia-fs method, so `false` is normal |
| `use_compat_mode : true` | If the GDS path fails, it processes via the CPU instead (default). Normal, but **this is why throughput alone cannot determine whether GDS is working** (step 4) |
| `force_compat_mode : false` | If `true`, it always goes through the CPU. `false` is normal |
| `fs.weka...`, `fs.gpfs...`, `block.nvme...`, etc. | Settings for other storage. Irrelevant |

**(d) GPU INFO**

| Line | Interpretation |
|---|---|
| `supports GDS` | Check ③. Must be present on every GPU |
| `bar size (MiB):...` | The size of the region that exposes GPU memory over PCIe. Reference value |
| `IOMMU State: Disabled` | Normal (GDS works when IOMMU is off) |

**(e) PLATFORM INFO**

| Line | Interpretation |
|---|---|
| `IOMMU: disabled` | Check ④ |
| `Nvidia Driver Info Status: Supported` | The GPU driver supports GDS |
| `Cuda Driver Version Installed: ...` | The CUDA version of the host driver |
| `Platform verification succeeded` | Check ⑤. The entire platform inspection passed |

!!! note "Actual output example — normal (as of 2026-10)"
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
     Platform: <GDS-capable GPU instance>, Arch: x86_64(Linux 7.0.0-1013-aws)
     Platform verification succeeded
    ```

### When abnormal

| Symptom | What to check |
|---|---|
| `DDN EXAScaler` only has `compat` (no `nvfs`) | Whether nvidia-fs is loaded (step 1) / whether Lustre is mounted (`df -hT`) / whether `--optimized-for-gds` was configured (step 2) |
| There is a GPU without `supports GDS` | Check that GPU's status with `nvidia-smi`, then share with AWS |
| `Platform verification failed` | Share the entire output with AWS |

## 4. Verify actual GDS I/O (`gdsio`)

Whether GDS actually works is judged by **comparing two counters before and after running gdsio in GDS mode**. Throughput output alone cannot distinguish it from compat mode (going through the CPU).

- **nvidia-fs counters** (`/proc/driver/nvidia-fs/stats`): whether I/O went directly into GPU memory (the GDS path)
- **EFA counters** (`lnetctl ... send_count / recv_count`): whether FSx traffic flowed over EFA

### Main gdsio options

| Option | Meaning |
|---|---|
| `-f <file>` | Test file path (under the FSx mount) |
| `-d <number>` | GPU number (one at a time) |
| `-w <count>` | Number of worker threads |
| `-s <size>` | File size (e.g. `10G`) |
| `-i <size>` | Size of a single I/O (e.g. `1M`) |
| `-x <number>` | **Transfer method.** `0` = GDS (GPU direct), `1` = via CPU |
| `-I <number>` | **I/O type.** `0` = read, `1` = write |

### Execution order

Replace `/fsx` with the actual FSx mount path. Since the write creates the test file, run the write first.

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

1. Record counters before running
2. GDS write → GDS read
3. Record counters after running (compare with 1)
4. Comparison CPU-via read
5. Record counters after the CPU run (compare with 3 → the GDS counters must stay the same)

### How to read the gdsio output

| Field | Meaning |
|---|---|
| `IoType` | `WRITE` write / `READ` read (the `-I` value) |
| `XferType` | `GPUD` = **requested** in GDS mode (`-x 0`), `CPUONLY` = via CPU (`-x 1`). **Even when it shows `GPUD`, it may have been processed via compat mode instead, so do not judge GDS operation from this alone** |
| `Threads` | Number of worker threads (`-w`) |
| `DataSetSize` | Amount actually processed / amount specified (KiB). Normal if they are nearly equal |
| `IOSize` | Size of a single I/O (`-i`) |
| `Throughput` | Throughput of this single gdsio process (GiB/s) |
| `Avg_Latency` | Average latency of a single I/O (µs) |
| `ops` | Number of I/O operations processed |
| `total_time` | Elapsed time (seconds) |

!!! note "Actual output example — normal (as of 2026-10)"
    ```
    IoType: WRITE XferType: GPUD Threads: 4 DataSetSize: 10285056/10485760(KiB) IOSize: 1024(KiB) Throughput: 3.202468 GiB/sec, Avg_Latency: 1218.884017 usecs ops: 10044 total_time 3.062823 secs
    IoType: READ XferType: GPUD Threads: 4 DataSetSize: 10383360/10485760(KiB) IOSize: 1024(KiB) Throughput: 3.189802 GiB/sec, Avg_Latency: 1224.218035 usecs ops: 10140 total_time 3.104376 secs
    IoType: READ XferType: CPUONLY Threads: 4 DataSetSize: 10434560/10485760(KiB) IOSize: 1024(KiB) Throughput: 3.706651 GiB/sec, Avg_Latency: 1052.802307 usecs ops: 10190 total_time 2.684680 secs
    ```

**GDS throughput not being higher than CPU does not mean GDS failed.** In a single-GPU, single-file test, the FSx file system side (the number of OSTs the file is stored on, the file system throughput limit) is often the bottleneck. The benefit of GDS appears in reduced CPU usage and in the aggregate bandwidth when multiple GPUs do I/O at the same time (see "Multi-GPU concurrent test" below).

### How to read the nvidia-fs counters

Compare `/proc/driver/nvidia-fs/stats` before and after running. The following three values are the key evidence of GDS path usage.

| Line | Meaning | Judgment |
|---|---|---|
| `Mmap` n / `Bar1-map` n | Number of times a GPU memory buffer was registered with nvidia-fs | Must increase after a GDS run (both write and read). Does not increase after a CPU run — **the key evidence of GDS path usage** |
| `Sparse Reads` n | Number of read I/Os processed by nvidia-fs | Must increase after a GDS read (e.g. 10 GiB ÷ 1 MiB = 10240). `holes=0` is normal |
| `err=`, `Error` lines | Error counts | Must all be 0 |
| `Reads` / `Writes` | Read/write error counts | With `IO stats: Disabled` (default), only error counts appear and operation counts do not. 0 is normal |
| `IO stats: Disabled` | Detailed read/write statistics are off | Default and normal. Can be judged with the three values above |
| `Mellanox PeerDirect Supported: False` | A feature only for Mellanox NICs | Normal in an EFA environment |
| `Active Process`, `Ops`, `Registered_MiB`, `Cache_MiB` | GDS work running **at this very moment** | 0 is normal after gdsio finishes. These are not cumulative, so do not use them for judgment |

!!! note "Actual output example — normal (as of 2026-10)"
    **Before running**
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

    **After running GDS write and read (10 GiB each) (changed lines only)**
    ```
    Sparse Reads  : n=10240 io=0 holes=0 pages=0
    Mmap          : n=12 ok=12 err=0 munmap=12
    Bar1-map      : n=12 ok=12 err=0 free=12 callbacks=0 active=0 delay-frees=0
    ```
    - `Mmap`/`Bar1-map`: 4 → 12 (+8). 2 GDS runs × 4 threads = 8. Evidence of GDS path usage.
    - `Sparse Reads`: 0 → 10240 (10 GiB ÷ 1 MiB). Evidence of the GDS read.

    **After the CPU-via (`-x 1`) run:** **no change** from the values above (going via the CPU does not go through nvidia-fs).

**Judgment:** if `Mmap`/`Bar1-map` increase after the GDS-mode run, `Sparse Reads` increases after the read, and they stay the same after the CPU run, then **the GDS path worked**. To see the write count directly as well, enable detailed statistics with `echo 1 | sudo tee /sys/module/nvidia_fs/parameters/rw_stats_enabled` (it is cleared on reboot) and then run again. It changes to `IO stats: Enabled` and the counts additionally appear on the `Reads`/`Writes` lines.

### How to read the EFA counters

The `send_count`/`recv_count` in `lnetctl net show --net efa -v` are the number of LNet messages that went over each EFA interface (not bytes). Compare before and after running to confirm that FSx traffic flowed over EFA.

| Item | Interpretation |
|---|---|
| Number of `send_count`/`recv_count` line pairs | Number of EFA interfaces. If it equals the baseline count, all are attached to LNet |
| There is an interface that increased greatly | **Evidence that FSx traffic flowed over EFA** |
| Only some of them increased greatly | **Normal.** Since the test used a single GPU (`-d 0`), the EFA interface close to that GPU on PCIe was used. If you use all GPUs at the same time, all interfaces increase (see below) |
| The rest increased slightly | Connection keep-alive messages. Normal |
| All interfaces stayed roughly the same | FSx traffic did not flow over EFA → inspect step 2 |

!!! note "Actual output example — normal (as of 2026-10, 8-EFA node)"
    After GDS write and read with a single GPU, only the 2 interfaces close to that GPU (out of 8) increased greatly:

    | EFA interface | send_count before → after | recv_count before → after |
    |---|---|---|
    | 1 | 53 → 56 | 53 → 56 |
    | 2 | 53 → 56 | 53 → 56 |
    | 3 | 53 → 55 | 53 → 55 |
    | 4 | 52 → 55 | 52 → 55 |
    | **5** | **2,063 → 17,442** | **4,111 → 24,611** |
    | **6** | **2,062 → 17,407** | **4,110 → 24,574** |
    | 7 | 14 → 20 | 14 → 20 |
    | 8 | 14 → 19 | 14 → 19 |

### (Optional) Multi-GPU concurrent test — confirm aggregate node throughput and full EFA usage

Since `-d` can specify only one GPU, run gdsio concurrently as many times as there are GPUs. Confirm that all EFA interfaces are used evenly and the aggregate throughput goes up. Replace `/fsx` with the actual mount path.

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

> Adjust the GPU number range in the loop above (`0 1 2 ... 7`) to match the number of GPUs on the node.

| Command/output | Interpretation |
|---|---|
| `lfs setstripe -c -1 /fsx/gds8` | Stripes newly created files in this directory across all OSTs. Prevents a single file being concentrated on a single OST and becoming a bottleneck. No effect on other paths |
| `cat /tmp/gds_*_*.log` | Per-GPU results. All must be `XferType: GPUD` with no errors |
| `WRITE total GiB/s:` / `READ total GiB/s:` | **The aggregate throughput of a single node.** gdsio has no aggregate value, so this is the sum of each `Throughput`. The upper bound is the FSx file system's throughput limit (capacity × throughput per TiB) |
| `paste` result | The left side is before running, the right side is after. **All interfaces must increase greatly** for EFA to have been used evenly across GPUs |

Cleanup:

```bash
rm -rf /fsx/gds8
```

### When the GDS path does not take: turn off compat mode and find the cause

If `Mmap`/`Bar1-map`/`Sparse Reads` stay the same even after a GDS run, then gdsio ran via the CPU (compat mode) even though it reported throughput. Turning off compat mode makes it raise an error instead of falling back to the CPU when the GDS path fails, so you can find the cause.

```bash
cp /etc/cufile.json /tmp/cufile.json
```

In `/tmp/cufile.json`, change the following value inside `"properties"`. Changing `"level"` inside `"logging"` in the same file to `"TRACE"` leaves detailed logs.

```json
"allow_compat_mode": false,
```

```bash
cd /tmp
CUFILE_ENV_PATH_JSON=/tmp/cufile.json /usr/local/cuda/gds/tools/gdsio -f /fsx/gds_test -d 0 -w 4 -s 1G -i 1M -x 0 -I 0
ls /tmp/cufile*.log
```

Since `CUFILE_ENV_PATH_JSON` points to the copy, the original `/etc/cufile.json` is not changed. Sharing the error message and `cufile.log` with AWS helps identify the cause.

### Clean up the test file

```bash
rm -f /fsx/gds_test
```

## 5. Check inside the container (Pyxis/Enroot)

Proceed with this after all the previous steps are normal on the host. Since user training code usually runs inside a container, verify that the host's nvidia-fs devices and GDS tools are passed through to the container.

```bash
srun -N1 --gres=gpu:8 --container-image=<training image> --container-mounts=/fsx:/fsx \
  bash -c 'ls -l /dev/nvidia-fs* | head -3; ls -d /usr/local/cuda*/gds/tools; grep "NVFS Driver" /proc/driver/nvidia-fs/stats'
```

> Match the `--gres=gpu:` value to the number of GPUs on the node.

| Output | Normal | Meaning |
|---|---|---|
| `ls -l /dev/nvidia-fs*` | A device file list appears | The host's nvidia-fs devices are passed through to the container |
| `ls -d .../gds/tools` | A path appears | The GDS tools (`gdscheck`, `gdsio`) are present inside the container |
| `NVFS Driver(version: ...)` | Same version as the host | It sees the host driver |

If these conditions are met, run step 3 (`gdscheck -p`) and step 4 (`gdsio`) inside the container the same way. Check the counters (`/proc/driver/nvidia-fs/stats`, `lnetctl`) **on the host**.

**If `/dev/nvidia-fs*` is not visible**, check whether the devices are passed through to the container in the Enroot settings (`ENROOT_RESTRICT_DEV`, etc.).

## 6. Result judgment summary

| Check | Normal | Meaning when abnormal | Action |
|---|---|---|---|
| `lsmod \| grep nvidia_fs` | Module visible | No GDS driver | Step 1 |
| `NVFS Driver(version)` | Minimum version or later | Version too low | Step 1 source build |
| `lnetctl ... grep -c "@efa"` | Baseline count | FSx not using EFA, or only partially | Step 2 `setup.sh --optimized-for-gds` |
| `gdscheck -p` `DDN EXAScaler` | Includes `nvfs` | No GDS path for Lustre | Step 3 "When abnormal" |
| `gdscheck -p` all GPUs | `supports GDS` | Problem with some GPUs | Share with AWS |
| `gdscheck -p` last line | `Platform verification succeeded` | Platform inspection failed | Share with AWS |
| `Mmap`/`Bar1-map` after GDS run | Increase | Ran in compat mode (via CPU) | "Turn off compat mode and find the cause" |
| `Sparse Reads` after GDS read | Increase | Reads did not take the GDS path | "Turn off compat mode and find the cause" |
| EFA `send_count`/`recv_count` after GDS run | Some increase greatly (all increase if all GPUs concurrent) | FSx traffic did not flow over EFA | Step 2 |
| `Unsupported` / `Disabled` / `false` entries | — | **Normal if they are entries listed in step 3** | None |

## 7. Information to share with AWS

If the problem is not resolved, please send the following output together.

1. `grep "NVFS Driver" /proc/driver/nvidia-fs/stats`
2. `sudo lnetctl net show`
3. `systemctl status configure-efa-fsx-lustre-client.service`
4. The full `gdscheck -p` output
5. The gdsio run results and the `/proc/driver/nvidia-fs/stats` and `lnetctl ... send_count|recv_count` values before and after running
6. (If you reproduced with compat mode off) the error message and `cufile.log`

## Reference documents

| Document | URL |
|---|---|
| FSx for Lustre — Configuring EFA clients | https://docs.aws.amazon.com/fsx/latest/LustreGuide/configure-efa-clients.html |
| FSx for Lustre — EFA file systems | https://docs.aws.amazon.com/fsx/latest/LustreGuide/efa-file-systems.html |
| NVIDIA GDS troubleshooting guide | https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/ |
| NVIDIA GDS configuration guide (cufile.json) | https://docs.nvidia.com/gpudirect-storage/configuration-guide/ |
