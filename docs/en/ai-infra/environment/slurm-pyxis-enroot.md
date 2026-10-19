---
title: Slurm + Pyxis + Enroot
description: Build the scheduler and container-runtime layers of a GPU training cluster. Covers Slurm and munge setup, node name resolution (including the IPv6 link-local pitfall), Enroot and Pyxis configuration, and how to submit containerized training jobs. Includes job submission and nccl-tests distributed-training validation (go/no-go).
---

# Slurm + Pyxis + Enroot

This page explains how to build the **scheduler and container-runtime layers** of a GPU training cluster. It assumes the host (driver, EFA, container toolkit, etc.) has already been validated, and layers the following on top of it.

- **Slurm** — the workload manager that groups nodes and schedules jobs
- **munge** — inter-node authentication (shared-key based)
- **Enroot** — an unprivileged runtime that runs container images without root privileges
- **Pyxis** — a Slurm SPANK plugin that acts as the glue, letting you launch an Enroot container with a single `srun` line

The goal is to launch a containerized training job across all nodes with a single `srun --container-image=...` line, **without SSH between nodes**. This layer has a number of pitfalls that quietly erode distributed training performance or fail jobs outright, so below we explain "why we do it this way" alongside each step.

!!! note "Validated environment (as of 2026-10)"
    The versions and values below are a reference snapshot validated at a specific point in time. Update them to match your actual configuration. The principles (body text) remain valid regardless of version.

    - Host OS: Ubuntu 24.04 (Deep Learning Base OSS Nvidia Driver GPU AMI)
    - Slurm: 23.11.4 (Ubuntu apt package)
    - Pyxis: 0.24.0
    - Enroot: 4.2.1
    - Instances: GPU nodes (8 GPUs per node, 8 EFA per node)
    - Scheduler topology: one GPU node doubles as the controller (`slurmctld`)
    - Node access: SSM Session Manager (no SSH between nodes)

## Architecture: controller-colocated configuration

For small- to medium-scale clusters, a configuration where **one GPU node doubles as the Slurm controller (`slurmctld`)** is simple and practical. You don't need a separate management node, and operations are simple because every node uses the same AMI and the same set of daemons.

```
[ GPU nodes × N (single block recommended) ]
  Node 1 : slurmctld (controller colocated) + slurmd + munge + Enroot + Pyxis
  Node 2 : slurmd + munge + Enroot + Pyxis
  ...
  Node N : slurmd + munge + Enroot + Pyxis

  Network : NVLink within a node / EFA between nodes
  Shared storage : /fsx (container image sqsh, data, checkpoints)
```

- `slurmd` runs on every node, and one of them additionally runs `slurmctld`.
- Inter-node authentication is handled by the munge shared key, so no SSH between nodes is needed. All distributed execution is carried out through `srun`.
- If you bake the Slurm, Pyxis, and Enroot setup into a single **operational AMI**, replacing a failed node is just a matter of launching the same template again.

!!! note "Validated environment (as of 2026-10)"
    The validated configuration used Slurm 23.11.4 from Ubuntu apt. Even if you build Slurm directly or install it from a different package in your actual environment, each of the items below (node name resolution, munge, the Enroot `/run` directory, Pyxis plugin registration) applies the same way.

## Step 1. Node name resolution (the most common pitfall)

Slurm has the controller and each `slurmd` find and communicate with each other **over IPv4**. However, on recent Ubuntu (24.04), **a node's own hostname can resolve to an IPv6 link-local address (`fe80::...`)**. In that state, Slurm nodes fall into `down` or communication errors occur.

First, check the current resolution result.

```bash
getent ahostsv4 $(hostname)   # whether it resolves over IPv4
getent hosts $(hostname)      # a problem if it resolves to fe80::... (IPv6 link-local)
```

If you see `fe80::...`, apply **one** of the following to ensure the hostname always resolves to IPv4.

- Specify `NodeAddr=<private IPv4>` in the node definition of `slurm.conf`. (A way to tell Slurm each node's communication address directly.)
- Add a `<private IPv4>  <hostname>` entry to `/etc/hosts` on all nodes.

Both approaches achieve the same goal of "making Slurm find nodes over IPv4." If you include this setting when building the operational AMI, you won't need to repeat it manually every time you replace a node.

## Step 2. munge (inter-node authentication)

munge is a shared-key mechanism that authenticates requests between Slurm daemons. **Every node must have the same `/etc/munge/munge.key`**, and the key file must be owned by `munge` with permissions `400`.

```bash
# (Once, on the controller) generate the key
sudo /usr/sbin/mungekey --verbose
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key
```

Since there is no SSH between nodes, distribute the generated key to all nodes **via the shared storage (`/fsx`)** and then copy it into each node's `/etc/munge/`. Once distribution is complete, be sure to delete the copy left on the shared location (to prevent key leakage).

```bash
# On each node
sudo install -o munge -g munge -m 400 /fsx/<path>/munge.key /etc/munge/munge.key
sudo systemctl enable --now munge
munge -n | unmunge | grep STATUS   # confirm STATUS: Success (0)
```

## Step 3. slurm.conf and gres.conf

These define Slurm's nodes and resources. The key points are (1) registering GPUs as GRES (Generic RESource), and (2) accurately describing each node's hardware topology. For node values you must use **exactly the values that `slurmd -C` outputs on each node** (if they differ, the node goes `down`).

```ini
SlurmctldHost=<controller>(<controller IPv4>)
AuthType=auth/munge
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
SelectType=select/cons_tres
SelectTypeParameters=CR_Core
GresTypes=gpu

# Common node defaults — matched to slurmd -C output
NodeName=DEFAULT CPUs=192 Boards=1 SocketsPerBoard=2 CoresPerSocket=48 \
  ThreadsPerCore=2 RealMemory=<slightly less than the slurmd -C value> Gres=gpu:8 State=UNKNOWN

# Per-node communication address (avoids the Step 1 IPv6 pitfall)
NodeName=<host-1> NodeAddr=<IPv4-1>
NodeName=<host-2> NodeAddr=<IPv4-2>
# ...

PartitionName=train Nodes=ALL Default=YES MaxTime=INFINITE State=UP
```

Register the GPU device files in `gres.conf`.

```ini
Name=gpu File=/dev/nvidia[0-7]
```

Things to keep in mind:

- Set **`RealMemory` slightly lower than the value reported by `slurmd -C`**. If the actual available memory is even slightly below the reported value, the node falls into `down`.
- If you run with `ConstrainDevices` set to `yes` in `cgroup.conf`, isolation ensures only the GPUs allocated to the job are visible. In that case, be sure to confirm that `/dev/infiniband` (EFA) and `/dev/gdrdrv` (GDRCopy) are **still visible** inside the job. If these get hidden, EFA communication and GPUDirect will not work.

!!! note "Validated environment (as of 2026-10)"
    - Node topology: `CPUs=192`, `SocketsPerBoard=2`, `CoresPerSocket=48`, `ThreadsPerCore=2`, `Gres=gpu:8` (based on `slurmd -C` output)
    - The validated configuration set `ConstrainDevices=no` in `cgroup.conf`. To run with `yes`, you need to confirm the device visibility described above.

## Step 4. PMIx (MPI launcher)

To launch MPI-based programs (e.g., `all_reduce_perf` from `nccl-tests`) without SSH between nodes, use Slurm's `--mpi=pmix`. Since `srun` coordinates each rank via PMIx, no SSH between hosts is needed.

```bash
srun --mpi=list   # pmix must appear in the list to use --mpi=pmix
```

- `--mpi=pmix` can only be used if Slurm was built with PMIx support. If you build Slurm directly, build it with `--with-pmix`.

!!! note "Validated environment (as of 2026-10)"
    Slurm 23.11.4 from Ubuntu apt is built with PMIx (v5) support, so `pmix` appears in `srun --mpi=list`.

## Step 5. Enroot (unprivileged container runtime)

Enroot runs containers without root privileges. The key configuration points are (1) **persisting the runtime directory `/run/enroot` across every boot**, and (2) placing large-capacity paths such as cache and data on **node-local NVMe**.

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

### Persisting `/run/enroot` with tmpfiles

`/run` is tmpfs, so **it is emptied on reboot.** If this directory is missing, a Pyxis job fails as follows.

```
mkdir: cannot create directory '/run/enroot': Permission denied
```

So register it with systemd-tmpfiles so that it is **created automatically on every boot**. (If you forget and create it manually just once, it disappears again after a node reboot.)

```bash
echo "d /run/enroot 1777 root root -" | sudo tee /etc/tmpfiles.d/enroot.conf
sudo systemd-tmpfiles --create /etc/tmpfiles.d/enroot.conf
```

Things to keep in mind:

- **Cache and data go on node-local NVMe (`/opt/dlami/nvme`).** Container images can be tens of GB, so placing them on the root EBS fills it up instantly.
- With **`ENROOT_RESTRICT_DEV n`**, `/dev/infiniband` (EFA) is visible inside the container as-is without needing a separate mount specification. It is required for EFA communication.

## Step 6. Pyxis (Slurm ↔ Enroot glue)

Pyxis is installed as a Slurm SPANK plugin. Once registered, options like `--container-image` appear on `srun`, and when a job launches, Pyxis automatically starts an Enroot container.

```bash
# Building requires the Slurm headers (libslurm-dev).
git clone https://github.com/NVIDIA/pyxis.git && cd pyxis && sudo make install

# Register the SPANK plugin
echo "required /usr/local/lib/slurm/spank_pyxis.so" | sudo tee /etc/slurm/plugstack.conf.d/pyxis.conf
echo "include /etc/slurm/plugstack.conf.d/*"        | sudo tee /etc/slurm/plugstack.conf

sudo systemctl restart slurmd
```

Verify registration:

```bash
srun --help | grep container-image   # success if the --container-image option appears
```

## Step 7. Basic operation check

Once you reach this point, the scheduler and container layers are complete. Check them in the following order.

```bash
sinfo                                   # whether nodes appear in the idle state
srun -N2 --ntasks-per-node=1 hostname   # inter-node execution (munge/PMIx path)

# Run a Pyxis + Enroot container (NGC image)
srun -N2 --ntasks-per-node=1 --gres=gpu:8 \
  --container-image=nvcr.io#nvidia/pytorch:<tag> \
  bash -c 'hostname; nvidia-smi -L | wc -l; ls /dev/infiniband | head -3'
```

- If `nvidia-smi -L` shows 8 lines per node and EFA devices appear under `/dev/infiniband`, the container has correctly picked up GPU and EFA.
- The registry notation for `--container-image` uses the `nvcr.io#nvidia/pytorch:<tag>` form (a Pyxis convention). When using a sqsh file on shared storage, specify it as `--container-image=/fsx/<path>/<image>.sqsh`.

!!! note "Validated environment (as of 2026-10)"
    The container image was validated with `nvcr.io/nvidia/pytorch:26.08-py3` (and an image with EFA installer 1.50.0 added to it). The NGC image already includes the AWS EFA stack, and 2-node container communication achieved the same performance as the host (but `--cpus-per-task` is mandatory — see below).

> 📎 [NVIDIA Pyxis](https://github.com/NVIDIA/pyxis) · [NVIDIA Enroot](https://github.com/NVIDIA/enroot)

## Submitting jobs

Once the scheduler and container layers are in place, actual training jobs are submitted with `sbatch`/`salloc` + `srun`. Below is the basic form of a containerized training job.

```bash
#!/bin/bash
#SBATCH -N <number of nodes>
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500

srun --container-image=/fsx/<path>/<image>.sqsh \
     --container-mounts=/fsx:/fsx \
     <training launch script>
```

Structure:

- Launch **8 ranks** (`--ntasks-per-node=8`) for the 8 GPUs per node, with each rank binding to one GPU.
- Request 8 GPUs per node with `--gres=gpu:8`.
- Do not assume consecutive numbering for the master node address; obtain it with **`scontrol show hostnames "$SLURM_JOB_NODELIST"`**.

### Two options you must always keep in mind

These two options are especially easy to miss because the job "runs" even without them. One quietly erodes performance, and one kills the job outright.

**1. Always specify `--cpus-per-task`**

This is the number of CPU cores to allocate per rank. If you don't specify it, depending on the Slurm configuration only **one CPU core per rank** may be allocated, and in that case communication performance drops to **a quarter to a tenth of the baseline**. EFA appears to be "running fine" (`Using network Libfabric`) while only the bandwidth comes out low — a hard-to-diagnose symptom. It happens because the network stack has insufficient CPU to process packets.

**2. When using `srun --export`, include `ALL`**

When passing environment variables with `srun --export`, always include `ALL` (`--export=ALL,VAR=value,...`). If you list only variables without `ALL`, the basic environment such as `PATH` is not passed, and the job **fails immediately** as follows.

```
slurmstepd: error: execve(): bash: No such file or directory
```

Supplementary rules:

- You may set them with `export` inside the job script and omit `--export` (the default is `ALL`).
- You **cannot use commas** inside an `--export` value (Slurm reads them as variable separators). For example, a value containing commas like `NCCL_DEBUG_SUBSYS=INIT,NET` should not be put in `--export`; set it with `export` in the job script instead.
- When running multiple `srun` invocations within one allocation (`salloc`/`sbatch`), reuse the container with `--container-name=<name>`, and from the second invocation onward it starts immediately without unpacking the sqsh again.

### Environment variables (EFA + NCCL)

NGC-family images already include the AWS EFA stack and the correct default environment variables. **For most variables, not specifying them is the correct answer**, and specifying them incorrectly causes a fallback to TCP or an initialization failure.

| Variable | Required? | Description |
|---|---|---|
| `NCCL_NET` | **No need to set** | The image default (`NCCL_NET_PLUGIN=ofi`) selects the EFA plugin. If you insist on setting it, it must be `Libfabric` (case-insensitive); setting `IB` fails initialization, and the older name (`AWS Libfabric`) produces no communication results. Do not use either value. |
| `NCCL_NET_PLUGIN` | Do not change | Keep the image default `ofi`. Changing it to another value points to the IB plugin and departs from the EFA path. |
| `FI_PROVIDER`, `FI_EFA_USE_DEVICE_RDMA` | No need to set | EFA and GPUDirect RDMA are used correctly even without setting them. Harmless if set. |
| `NCCL_DEBUG=INFO` | For inspection | Handy for confirming `Using network Libfabric` and `found 8 nics` in the job startup logs. |
| IB-environment `NCCL_IB_*`, `NCCL_NET=IB`, etc. | **Remove** | Not used in an EFA environment. If IB settings remain, they depart from the EFA path, drop to TCP, or fail initialization. |

- Key principle: **In an EFA environment, the default is to "not touch" NCCL network-related variables**. If you use `NCCL_*` settings carried over from a different (IB) environment as-is, performance drops or the job fails.
- During inspection, if the `NCCL_DEBUG=INFO` logs show `Using network Libfabric` and `provider is efa (... found 8 nics)`, you are on the EFA path. If you see `Using network Socket`, it has fallen back to TCP (sharp performance drop), so check whether `NCCL_NET_PLUGIN=ofi` and whether leftover IB settings remain.

!!! note "Validated environment (as of 2026-10)"
    - Operation was confirmed with aws-ofi-nccl 1.19.0 (bundled with NGC 26.08) and 1.21.1 (EFA installer 1.50.0), and **1.21.1 is recommended**.
    - Measured performance drop when `--cpus-per-task` was not specified (8 GiB): all-reduce algbw 92 → 368 GB/s, all-to-all 8.7 → 95 GB/s (restored when `--cpus-per-task=24` was applied).
    - In both versions above, the EFA plugin is selected even without setting `NCCL_NET`.

> 📎 [How to use Pyxis](https://github.com/NVIDIA/pyxis/blob/main/README.md) · [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

## Troubleshooting summary

These are the symptoms and remedies you actually encounter most often in this layer.

| Symptom | Cause | Remedy |
|---|---|---|
| Slurm node is `down`, communication errors, `getent hosts $(hostname)` returns `fe80::...` | Own hostname resolves to IPv6 link-local | Specify `NodeAddr` (IPv4) in `slurm.conf` or add to `/etc/hosts` (Step 1) |
| Node is `down`, `RealMemory`-related | Actual available memory < configured value | Set `RealMemory` slightly lower than the `slurmd -C` value (Step 3) |
| Pyxis job shows `mkdir: cannot create directory '/run/enroot': Permission denied` | `/run/enroot` disappeared on reboot | Ensure creation with tmpfiles (Step 5) |
| `slurmstepd: error: execve(): bash: No such file or directory` | `ALL` missing from `--export` | `--export=ALL,...` (job submission) |
| EFA appears to run (`Using network Libfabric`) but communication is a quarter to a tenth of the baseline | 1 CPU core per rank | `--cpus-per-task=24` (job submission) |
| NCCL logs show `Using network Socket` | EFA plugin not loaded / TCP fallback | Check whether `NCCL_NET_PLUGIN` is `ofi` and whether IB `NCCL_*` settings remain |
| Communication initialization fails with `NCCL_NET=IB`, etc. | IB path forced on an EFA environment | Remove IB settings (job submission) |
| `/dev/infiniband` not visible inside the container | `cgroup.conf` device isolation or `ENROOT_RESTRICT_DEV` setting | Confirm device visibility when using `ConstrainDevices` (Step 3), `ENROOT_RESTRICT_DEV n` (Step 5) |

## Distributed training validation (nccl-tests) { #distributed-training-validation }

Once you have finished configuring all of the cluster's nodes, network, and containers, you must **always** verify, **before** starting actual training, that the collective communication performance between nodes reaches the expected level. The section below is that **final validation gate**. If you cannot pass here (no-go), then even if you start training, the network will become a bottleneck and performance will not materialize, or training will slow down abnormally.

This step lets you do the following.

- Measure 2-node all-reduce / all-to-all bandwidth (host nccl-tests)
- Measure inter-node latency (`fi_pingpong`)
- Measure communication from inside a container (the training image) (PyTorch-based, avoiding MPI conflicts)
- How to read nccl-tests logs to determine whether the EFA path is used and its performance

### Why this gate is needed

In large-scale GPU training, each step consists of **GPU computation + inter-node communication**. Within a node (8 GPUs) they are connected by NVLink, and between nodes they are connected by EFA. The inter-node EFA bandwidth is about 3.2 Tbps per instance (about 400 GB/s), which divided across 8 GPUs is **about 50 GB/s per GPU** — a large difference from NVLink (several hundred GB/s per GPU in one direction). In other words, as the number of nodes grows, inter-node communication dominates overall throughput, so you must verify before training whether that path **operates close to the designed bandwidth**.

Another important reason for this validation is that most causes of degraded communication performance are **configuration issues that do not surface until you actually run training**. Representatively:

- NCCL fails to pick up the EFA plugin and falls back to TCP (Socket) → performance drops to a fraction
- Only 1 CPU core is assigned per rank, so communication performance degrades to 1/4–1/10
- A missing security group self-reference rule causes EFA communication itself to fail
- IB-related environment variables remain, so EFA is not selected

If you secure **baseline** numbers in this gate, then later during operations, when performance drops, you can compare against "what did this environment originally produce." For that reason, you must always record the measured values.

### Validation sequence (recommended)

Validate the communication path by narrowing it down one step at a time, from the bottom → up. If there is a problem at a lower step, the higher steps are meaningless, so always proceed in order.

1. **Host EFA·NCCL baseline** — Measure all-reduce·all-to-all across 2 or more nodes using the DLAMI's built-in nccl-tests (Slurm `srun --mpi=pmix`)
2. **Container NCCL path** — Run the PyTorch measurement script with the training image and verify it is similar to the host (`--cpus-per-task` required)
3. **Inter-node latency** — Measure the round-trip latency of small messages with `fi_pingpong`
4. **(Optional) EP dispatch check** — If using MoE, a standalone test of the Expert Parallelism backend
5. **Short training with the target model·parallelization configuration** — Run only a few dozen iterations and record the iteration time as a baseline
6. **(If the capacity blocks are split into several)** After checking the topology, measure within a block and between blocks separately and compare

The section below focuses on steps 1–4 (pure communication validation).

### 1. Host 2-node nccl-tests (EFA baseline)

First, run nccl-tests **on the host without a container** to verify that the EFA·NCCL path is healthy. The host has the driver·EFA stack·nccl-tests preinstalled, so you can obtain a "pure" baseline that excludes container variables.

- nccl-tests is installed per CUDA version. Run it with 8 ranks per node and 1 GPU per rank (`-g 1`).
- Run with `srun --mpi=pmix` without inter-node SSH. This is only possible when Slurm is built with PMIx support — first check whether `pmix` appears in `srun --mpi=list`.
- **Always specify `--cpus-per-task`.** If you do not, only 1 CPU core may be assigned per rank, and in that case communication performance drops significantly.

!!! note "Validated environment (as of 2026-10)"
    The paths·versions·numbers below are a **reference** measured at a specific point in time on a P6-B200 2-node environment. If the versions change, you only need to update this box.

    - DLAMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)
    - nccl-tests path: `/usr/local/cuda-XX/efa/test-cuda-XX/` (provided for each of CUDA 12.8·12.9·13.0·13.2)
    - Driver-aligned CUDA 13.2, EFA installer 1.50.0 (libfabric 2.6.0, aws-ofi-nccl 1.21.1, PMIx 4.2.8)
    - Slurm 23.11.4 (PMIx v5 support), Pyxis 0.24.0, Enroot 4.2.1
    - In the example commands below, change `cuda-13.2` to match the actually installed CUDA version.

```bash
TESTDIR=/usr/local/cuda-13.2/efa/test-cuda-13.2
LIBS=/usr/local/cuda-13.2/lib:/usr/local/cuda-13.2/lib64:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib

# all-reduce (data-parallel communication pattern)
srun -N2 --ntasks-per-node=8 --cpus-per-task=24 --gres=gpu:8 --mpi=pmix \
  --export=ALL,LD_LIBRARY_PATH=$LIBS,NCCL_DEBUG=INFO \
  $TESTDIR/all_reduce_perf -b 8 -e 16G -f 2 -g 1 -c 1 -n 50

# all-to-all (MoE expert-parallel communication pattern)
srun -N2 --ntasks-per-node=8 --cpus-per-task=24 --gres=gpu:8 --mpi=pmix \
  --export=ALL,LD_LIBRARY_PATH=$LIBS,NCCL_DEBUG=INFO \
  $TESTDIR/alltoall_perf -b 8 -e 16G -f 2 -g 1 -c 1 -n 50
```

Flag meanings:

- `-b 8 -e 16G` : Sweep the message size from 8 bytes to 16 GiB
- `-f 2` : Increase the size by 2x each step
- `-g 1` : 1 GPU per rank (ranks are already 8 per node via `--ntasks-per-node=8`)
- `-c 1` : Perform a correctness check
- `-n 50` : Measure 50 iterations per size

!!! tip "Always include `ALL` in `--export`"
    If you list only variables in `srun --export`, things like `PATH` are not passed and the job can fail immediately. Always use the form `--export=ALL,VAR=value,...`, or set them with `export` in a job script and omit `--export` (the default is `ALL`). Also, you cannot use commas inside the `--export` value (Slurm reads them as variable separators).

#### Single-node pre-check (NVLink only)

Before moving to 2 nodes, first verifying with a 1-node all-reduce that the GPU·driver·NCCL itself is healthy helps narrow the scope of the problem. A single node uses only in-node NVLink, so it checks the health of the GPU stack independently of EFA.

```bash
TESTDIR=/usr/local/cuda-13.2/efa/test-cuda-13.2
/opt/amazon/openmpi/bin/mpirun -n 8 --bind-to none \
  -x LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib:/usr/local/cuda-13.2/lib64:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH \
  -x NCCL_DEBUG=INFO \
  $TESTDIR/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 50
```

### 2. Container 2-node communication (PyTorch measurement)

Once the host baseline is confirmed, validate that the same performance is produced **inside the container image you will use for actual training**. This is because the EFA stack·NCCL·plugin settings inside the image may differ from the host.

!!! warning "Do not run nccl-tests via MPI inside the container"
    NGC-family training images bundle their own Open MPI. If you run a program built with the EFA installer's Open MPI (including nccl-tests) inside the container via MPI, the two Open MPI installations mix and it fails with `opal_shmem_base_select failed` → `MPI_INIT failed`. Therefore, **the in-container communication check is done with a PyTorch-based measurement that does not require MPI**. This script picks up the rank from Slurm environment variables (such as `SLURM_PROCID`) and initializes the distributed group without MPI.

Place the measurement script in shared storage and run it with the training image.

```bash
salloc -N2 --ntasks-per-node=8 --cpus-per-task=24 --gres=gpu:8
srun --container-image=/<shared-path>/<image>.sqsh --container-name=train --container-mounts=/<shared-path>:/<shared-path> \
  --export=ALL,MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1),MASTER_PORT=29500,NCCL_DEBUG=INFO \
  python /<shared-path>/torch_comm_bench.py
```

!!! info "Reference measurement script — `torch_comm_bench.py`"
    A PyTorch measurement script that picks up the rank from Slurm environment variables without MPI. The busbw formula is the same as nccl-tests. This is a reference example for a specific environment, so adjust the library versions·measurement ranges to match your actual environment.

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

!!! warning "Assign enough CPUs per rank"
    If you do not specify `--cpus-per-task`, only 1 CPU core may be assigned per rank, and in that case, even if EFA operates normally, communication performance **drops to 1/4–1/10**. There is a case where, in a single measurement, 8 GiB all-reduce changed from algbw 92 → 368 GB/s and all-to-all from 8.7 → 95 GB/s. Specify the instance's vCPU divided by the number of GPUs (e.g., 192 vCPU ÷ 8 GPU = 24).

### 3. Inter-node latency (`fi_pingpong`)

Separately from bandwidth, also measure the **round-trip latency** of small messages. all-reduce is sensitive to bandwidth, but communication patterns with many small messages are sensitive to latency.

```bash
salloc -N2 --ntasks-per-node=1
srun --overlap -N1 -n1 -w <node-A> /opt/amazon/efa/bin/fi_pingpong -p efa -e rdm -I 10000 -S 64 &
sleep 5
srun --overlap -N1 -n1 -w <node-B> /opt/amazon/efa/bin/fi_pingpong -p efa -e rdm -I 10000 -S 64 <node-A IP>
wait
```

- The default endpoint of `fi_pingpong` is dgram, so specify `-e rdm`.
- `-I` is the number of iterations, and `-S` is the message size (bytes).
- Start the server side first (`&`), wait briefly, then connect from the client side by specifying the server IP.

### 4. Baseline values and decision

When measurement is done, compare against the baseline values below and **record the measured values from your actual environment as the baseline**.

!!! note "Validated environment (as of 2026-10) — reference baseline values"
    The numbers below are **reference values measured at a specific point in time on a P6-B200 2-node environment, and are not guaranteed-performance values**. They vary by number of nodes, model, parallelization configuration, and software versions, so treat the measured values from your actual environment as the baseline.

    | Measurement | Reference baseline value |
    |---|---|
    | Host all-reduce (16 GiB) | algbw 374 GB/s, busbw 702 GB/s |
    | Host all-to-all (16 GiB) | algbw 95 GB/s, busbw 89 GB/s |
    | Container all-reduce / all-to-all (8 GiB, PyTorch) | algbw 367–368 / 95 GB/s |
    | `fi_pingpong` 64 B | 10.85 µs/xfer |

    - For reference, the per-node EFA bandwidth is about 400 GB/s, and the all-reduce algbw of 374 GB/s above is about 94% of that.
    - all-to-all algbw of 95 GB/s, when converted to per-GPU inter-node traffic, is about 48 GB/s, which is about 95% of the per-GPU EFA ceiling (about 50 GB/s).

Decision guidance:

- **If the values drop significantly when you increase the number of nodes**, split the nodes in half and measure again to narrow down the problem nodes. After terminating a problem node, re-launch it with the same launch template and replace it.
- If the capacity blocks are split into several, measure a node pair within a block and a node pair between blocks separately and compare. Between blocks, the network segment (spine) differs, so performance may differ.

### 5. How to read nccl-tests results

After running with `NCCL_DEBUG=INFO` turned on, check in the following order.

#### ① Was the computation correct — error count

```
# Out of bounds values : 0 OK
```

This must be 0. The `#wrong` column of the result table must all be 0 as well. **If it is not 0, it is a failure regardless of performance** (no-go).

#### ② Which network was ridden — the front of the log

```
NET/OFI Initializing aws-ofi-nccl <version>
NET/OFI Selected provider is efa, fabric is efa-direct (found 8 nics)
Using network Libfabric
```

- `Using network Libfabric` and `provider is efa` must appear for it to be the EFA path.
- `found 8 nics` (8 EFA devices) must appear for all network cards to be used.
- If `Using network Socket` appears, it has fallen back to TCP, and performance drops significantly. In that case, refer to the troubleshooting below.
- GPUDirect RDMA usage is confirmed by `via NET/Libfabric/.../GDRDMA` in the log.

#### ③ Performance — the `algbw`·`busbw` of the result table

```
      size    count  type  redop root   time   algbw   busbw  #wrong | (in-place) ...
```

| Column | Meaning |
|---|---|
| size | The size of data sent at once (bytes) |
| time | Elapsed time (microseconds) |
| algbw | Data size ÷ time (algorithm bandwidth) |
| **busbw** | The value corrected from algbw to fit the communication pattern. A metric comparable to hardware bandwidth |
| left / right | out-of-place / in-place results (usually similar) |

- Look at the **large-message (several hundred MiB–GiB) rows**. For small messages, it is normal for bandwidth to come out low because of latency time.
- The last `Avg bus bandwidth` is the average across all sizes and is for reference only.

#### ④ What to compare against

- **1 node:** Uses only in-node NVLink. Its purpose is to check that the GPU·driver·NCCL are healthy.
- **2 or more nodes:** It rides EFA. In ②, first confirm it is the EFA path, then look at performance.
  - For all-reduce, if NCCL uses the NVLS tree (in-node reduction over the NVLink Switch, inter-node over EFA), busbw can come out similar to a single node. In that case, judge by **how close algbw is to the per-node EFA bandwidth**.
  - For all-to-all, the proportion of inter-node traffic is large, so it is normal for it to come out much lower. Compare the per-GPU inter-node traffic (half of algbw for 2 nodes) against the per-GPU EFA ceiling (about 50 GB/s).
  - Compare against the baseline values (section 4), and record the measured values from your actual environment as the baseline.

### Troubleshooting

| Symptom | Cause | Remedy |
|---|---|---|
| `Using network Socket` in the NCCL log | EFA plugin not loaded (TCP fallback) | Check whether the image's NCCL network plugin is for EFA (ofi) and whether IB-related `NCCL_*` settings remain |
| Communication initialization fails with `NCCL_NET=IB` etc. | The IB path is being forced | Remove IB-related environment variables (`NCCL_NET=IB`, `NCCL_IB_*`). In an EFA environment, not specifying `NCCL_NET` is safe |
| EFA is used (`Libfabric`) but performance is 1/4–1/10 of the baseline | 1 CPU core per rank | Set `--cpus-per-task` to vCPU ÷ number of GPUs |
| `slurmstepd: error: execve(): bash: No such file or directory` | No `ALL` in `--export` | Fix to `--export=ALL,...` |
| An MPI program inside the container shows `opal_shmem_base_select failed` | The image's Open MPI mixes with a different Open MPI | Perform the in-container communication check with the MPI-free PyTorch measurement |
| `--mpi=pmix` does not work | Slurm is not built with PMIx support | Check PMIx support with `srun --mpi=list` for `pmix`. If building Slurm directly, use `--with-pmix` |
| A device other than EFA appears in `/sys/class/infiniband/` or the container's `/dev/infiniband` | On some instances, an additional RDMA device in link-DOWN state appears (separate from EFA) | No action needed. It is fine if NCCL selects EFA (`Using network Libfabric`, `found 8 nics`). Do not force the IB path |
| `#wrong` is not 0 / `Out of bounds values` is not 0 | Data correctness failure | no-go regardless of performance. Inspect hardware·driver·memory |

### Checklist

- [ ] Confirm `pmix` support in `srun --mpi=list`
- [ ] 1-node all-reduce normal (GPU·driver·NCCL health)
- [ ] Host 2-node all-reduce·all-to-all measured (`--cpus-per-task` specified)
- [ ] Confirm `Using network Libfabric`, `found 8 nics` in the NCCL log (EFA path)
- [ ] `#wrong` / `Out of bounds values` both 0
- [ ] Container (training image) 2-node measurement is similar to the host (PyTorch measurement)
- [ ] `fi_pingpong` inter-node latency measured
- [ ] Record the measured values from your actual environment as the baseline

> 📎 Reference official docs: [Get started with EFA and NCCL](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html) · [nccl-tests](https://github.com/NVIDIA/nccl-tests) · [`fi_pingpong(1)`](https://ofiwg.github.io/libfabric/main/man/fi_pingpong.1.html) · [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

## Checklist

- [ ] Node hostname resolves to IPv4 — `NodeAddr`/`/etc/hosts` (Step 1)
- [ ] All nodes have the same `/etc/munge/munge.key` (owner munge, permissions 400), shared copy deleted (Step 2)
- [ ] `slurm.conf` node values match `slurmd -C`, GPU registered in `gres.conf` (Step 3)
- [ ] `pmix` appears in `srun --mpi=list` (Step 4)
- [ ] `/etc/enroot/enroot.conf` configured, cache and data on local NVMe, `/run/enroot` tmpfiles registered (Step 5)
- [ ] Pyxis SPANK plugin registered, `--container-image` appears in `srun --help` (Step 6)
- [ ] 2-node container run confirms 8 GPUs and `/dev/infiniband` (Step 7)
- [ ] Job submission: specify `--cpus-per-task`, `--export=ALL,...`, `--gres=gpu:8` (job submission)
- [ ] `NCCL_DEBUG=INFO` logs show `Using network Libfabric` and `found 8 nics` (job submission)
