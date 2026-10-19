---
title: Container Image (NGC + EFA)
description: How to build a container image for large-scale GPU training by adding the AWS EFA stack on top of an NGC PyTorch base image — focused on image-build pitfalls and the EFA communication stack.
---

# Container Image (NGC + EFA)

In a large-scale GPU training cluster, every node must share an **identical software environment** to guarantee reproducibility and performance. Instead of installing this environment directly on the host OS, you take the PyTorch container image distributed by NVIDIA NGC as a base and add the AWS EFA (Elastic Fabric Adapter) communication stack on top of it to build the training image. Once the resulting image is placed on shared storage as a `.sqsh` archive, each node pulls it out and runs the training job (follow your orchestrator's guide for how to launch jobs with the scheduler).

This page covers the following.

- The EFA stack already included in the NGC PyTorch image and its limitations
- The Dockerfile pattern and pitfalls of adding the EFA installer on top of the NGC base
- The container-specific EFA installer flags, `LD_LIBRARY_PATH`, and how to avoid MPI conflicts
- How to import the image as an Enroot `.sqsh` and reuse it on nodes

!!! note "Validated environment (as of 2026-10)"
    The versions and values below are a **reference snapshot** validated at a specific point in time on 2 P6-B200 nodes. The principles remain valid regardless of version; when versions change, you only need to update this box.

    - Base image: `nvcr.io/nvidia/pytorch:26.08-py3` (NCCL 2.30.7 · CUDA 13.4.1 · PyTorch 2.14)
    - Added EFA installer: 1.50.0 (libfabric 2.6.0, aws-ofi-nccl 1.21.1)
    - Scheduler: Slurm 23.11.4 · Pyxis 0.24.0 · Enroot 4.2.1
    - Host AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04), driver 595.91.07 / CUDA 13.2
    - Validated image size: about 25 GB (`.sqsh`)

## Why an NGC base

The NGC PyTorch image is built and distributed so that CUDA, cuDNN, NCCL, and PyTorch are mutually compatible. This saves you the trouble of matching the version combination yourself, and NVIDIA pre-applies optimizations tailored to the B-series GPUs.

Even if the container's CUDA version is higher than the host driver, it works through **CUDA Forward Compatibility mode**. When the container starts, `CUDA Forward Compatibility mode ENABLED` is displayed, and GPU computation, NCCL/EFA communication, and training all work correctly. However, if your training code contains custom CUDA kernels, it is safer to separately verify their behavior in that environment.

## The EFA stack already included in the NGC image

Recent NGC PyTorch images already include the AWS EFA stack. The configuration confirmed in the validated image is as follows.

| Item | Content |
|---|---|
| libfabric | Included in `/opt/amazon/efa` (2.4.0 at validation) |
| aws-ofi-nccl | `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so` (1.19.0 at validation) |
| Default environment variables | `NCCL_NET_PLUGIN=ofi`, `NCCL_TUNER_PLUGIN=ofi` |
| Library path | Registered in `/etc/ld.so.conf.d/`, so it is found without setting `LD_LIBRARY_PATH` separately |

The core principles are as follows.

- **The original image delivers normal EFA performance as-is.** You do not need to mount the host's EFA libraries or `/dev/infiniband` with `--container-mounts`. The EFA stack inside the image uses the host EFA devices directly.
- The approach of mounting host libraries into the container and inserting them into `LD_LIBRARY_PATH` is **not recommended.** If the versions of the in-image EFA stack and the host libraries are mixed, things can break. To align versions, use the image build approach described below rather than host mounts.
- **Do not remove or change `NCCL_NET_PLUGIN=ofi`.** The default-named `libnccl-net.so` points to a different plugin (for InfiniBand) inside the image. You must specify `ofi` for the EFA path to be selected.

!!! warning "The EFA version inside the image may differ from the host"
    The EFA version in the original NGC image may differ from the EFA installer version installed on the host AMI. Even if performance is fine, aligning versions makes troubleshooting simpler and lets you use components that require a specific minimum EFA installer version—such as communication libraries for EP (Expert Parallelism) scaling. For this reason, the build approach below is recommended.

## Recommended: add the EFA installer to the training image

For the training image, we recommend installing the **same version of the EFA installer** as the host AMI. Performance is the same as the original, but it matches the host versions (libfabric, aws-ofi-nccl) and satisfies the prerequisites (a specific minimum EFA installer version) of communication libraries for EP scaling.

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

### Pitfall ① — Run `apt-get update` first within the same RUN

**Before running the EFA installer, you must run `apt-get update` first within the same `RUN` layer.** The EFA installer installs third-party dependency packages via apt during execution. If the apt package list is empty, it fails as follows.

```
E: Unable to locate package environment-modules
...
failed to install third-party dependencies
```

Because Docker layers are cached, even if you run `apt-get update` in a separate `RUN`, the list may be empty in the layer where the EFA installer runs. Therefore it is safe to **combine `apt-get update` and the EFA installer run into a single `RUN`.**

### Pitfall ② — Container-specific EFA installer flags

Inside a container there is neither a kernel nor the actual EFA devices. Running the same installation process as on the host would fail, so you specify container-specific flags.

| Flag | Meaning | Reason |
|---|---|---|
| `--skip-kmod` | Skip kernel module installation | Kernel modules are the host's responsibility. Inside the container they cannot/need not be installed |
| `--skip-limit-conf` | Skip `limits.conf` (ulimit) modification | ulimits such as memlock are set on the host |
| `--no-verify` | Skip device verification after install | There are no EFA devices in the build environment |
| `-g` | Include GPUDirect-related configuration | RDMA path between GPUs/across nodes |

- The EFA installer detects the NGC base and installs the **NGC-specific aws-ofi-nccl package** (e.g., `libnccl-ofi-ngc-v3`). Therefore `--disable-ngc` is not needed.

### Pitfall ③ — `LD_LIBRARY_PATH`

To ensure the libraries installed by the EFA installer are found at runtime, place `/opt/amazon/ofi-nccl/lib` and `/opt/amazon/efa/lib` at the front of `LD_LIBRARY_PATH`.

```dockerfile
ENV LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}
```

This way the in-image EFA/aws-ofi-nccl is used preferentially, avoiding mix-ups with the older-version libraries originally present in the base image.

### Pitfall ④ — Do not layer another MPI on top of NGC Open MPI

The NGC image already includes Open MPI 5 (`/usr/local/mpi`). If you run a **program built with a different Open MPI** brought in by the EFA installer inside the container, the two Open MPIs get mixed up and fail as follows.

```
opal_shmem_base_select failed
--> Returned value -1 instead of OPAL_SUCCESS
...
MPI_INIT failed
```

**Principle: Leave the NGC image's MPI as-is and do not layer another Open MPI on top.** When verifying inter-node communication inside a container, we recommend using a **PyTorch-based measurement that does not need MPI** instead of a benchmark that relies on MPI. (`nccl-tests` on the host uses the host Open MPI and `srun --mpi=pmix`, so it is unrelated to this issue.)

!!! note "Validated environment (as of 2026-10) — full reference Dockerfile"
    Below is a full reference example that organizes the validated configuration. On top of stage 1 (EFA), it layers an MoE communication library built for intra-node use (stage 2) and the user training code (stage 3). Take only the stages you need for your actual training image. Point-in-time values such as commit hashes are kept only in this box.

    ```dockerfile
    FROM nvcr.io/nvidia/pytorch:26.08-py3

    ARG EFA_INSTALLER_VERSION=1.50.0
    ARG HYBRID_EP_COMMIT=94a9f8f6b146c07d97ec58f67cd6d303296d6098

    # 1. EFA installer (container-specific: exclude kernel modules). Run apt-get update first in the same RUN.
    RUN apt-get update \
     && apt-get install -y --no-install-recommends curl ca-certificates git \
     && cd /tmp \
     && curl -fsSL -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
     && tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
     && cd aws-efa-installer \
     && ./efa_installer.sh -y -g --skip-kmod --skip-limit-conf --no-verify \
     && cd / && rm -rf /tmp/aws-efa-installer* /var/lib/apt/lists/*

    ENV LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}

    # 2. MoE communication library (intra-node build only, B200)
    ENV TORCH_CUDA_ARCH_LIST=10.0
    RUN git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP \
     && cd /opt/DeepEP && git checkout ${HYBRID_EP_COMMIT} \
     && DISABLE_AGGRESSIVE_PTX_INSTRS=1 pip install --no-build-isolation --no-deps . \
     && rm -rf /opt/DeepEP/build

    # 3. Add your user training code (training framework fork, custom kernels, etc.) below this
    ```

## Building the image and importing it into Enroot

After building the image, convert it into a `.sqsh` archive with Enroot and place it on shared storage.

```bash
sudo docker build -t <image>:<tag> .
sudo enroot import -o /<shared-path>/<path>/<image>.sqsh dockerd://<image>:<tag>
```

### Why place the .sqsh on shared storage

- If you place the `.sqsh` file on shared storage and specify it with `--container-image=/<shared-path>/<path>/<image>.sqsh`, **each node does not need to pull directly from a registry (such as NGC).** When dozens of nodes pull the image from the registry simultaneously, you avoid the load and failure risk that would otherwise arise.
- Each time a job starts, Enroot unpacks the `.sqsh` onto the node's local NVMe. When running multiple `srun`s within a single allocation (`salloc`/`sbatch`), if you reuse the container with `--container-name=<name>`, from the second `srun` onward it starts immediately without unpacking again.

```bash
salloc -N <number of nodes> --ntasks-per-node=8 --cpus-per-task=24 --gres=gpu:8
# First srun: unpack .sqsh onto local NVMe and create container
srun --container-image=/<shared-path>/<path>/<image>.sqsh --container-name=train \
     --container-mounts=/<shared-path>:/<shared-path> bash -c 'nvidia-smi -L | wc -l'
# From the second srun: reuse by name → starts immediately
srun --container-name=train --container-mounts=/<shared-path>:/<shared-path> python /<shared-path>/<path>/train.py
```

### Letting Enroot expose EFA devices to the container

If the host Enroot setting is `ENROOT_RESTRICT_DEV n`, then inside the container `/dev/infiniband` (EFA) is visible as-is without a separate mount. Therefore you do not need to manually mount the device with `--container-mounts` for EFA communication. (For the host-side Enroot/Pyxis settings, refer to the scheduler configuration page.)

## Verifying operation

Do a quick check that the image comes up correctly on the nodes and that the GPU and EFA devices are visible. The example below uses Slurm + Pyxis; on another orchestrator, bring the container up the corresponding way.

```bash
srun -N2 --ntasks-per-node=1 --gres=gpu:8 \
  --container-image=/<shared-path>/<image>.sqsh \
  bash -c 'hostname; nvidia-smi -L | wc -l; ls /dev/infiniband | head -3'
```

Communication performance is measured in a separate validation step — inside the container with a **PyTorch-based measurement** (no MPI needed); for the full procedure see [Distributed training validation](slurm-pyxis-enroot.md#distributed-training-validation). If the following appears in the NCCL log (`NCCL_DEBUG=INFO`), the EFA path is in use.

```
NET/OFI Selected provider is efa, fabric is efa-direct (found 8 nics)
Using network Libfabric
```

- `Using network Libfabric` + `provider is efa` must appear for the EFA path.
- If `Using network Socket` appears, it has fallen back to TCP, and performance drops significantly. Check that `NCCL_NET_PLUGIN=ofi` is retained and that no InfiniBand-specific `NCCL_*` settings remain.
- You should see `found N nics` equal to the number of EFA devices on the instance (for example, `found 8 nics` on an instance with 8 EFAs).

!!! tip "If performance is far below the baseline"
    Even on the EFA path (`Using network Libfabric`), if communication bandwidth is 1/4 to 1/10 of the baseline, it is often because there are too few CPU cores per rank. Allocate enough cores per rank, such as `--cpus-per-task=24`. In the validated environment, `--cpus-per-task=24` was required when measuring with 2 container nodes.

## Summary of common issues

| Symptom | Cause | Remedy |
|---|---|---|
| `E: Unable to locate package environment-modules` during `docker build` | apt list is empty before the EFA installer runs | Run `apt-get update` first in the same `RUN` |
| An MPI program inside the container hits `opal_shmem_base_select failed` → `MPI_INIT failed` | NGC Open MPI mixed with a different Open MPI | Do not layer another Open MPI on the NGC image. Use a PyTorch-based measurement to verify communication |
| `Using network Socket` in the NCCL log | EFA plugin not loaded (TCP fallback) | Confirm `NCCL_NET_PLUGIN=ofi` is retained, remove IB-specific `NCCL_*` settings |
| EFA is active (`Libfabric`) but bandwidth is 1/4 to 1/10 of the baseline | Too few CPU cores per rank | Increase `--cpus-per-task` (e.g., 24) |
| The in-image EFA version differs from the host | Using the EFA built into the original NGC | Install the same EFA installer as the host in the training image to align versions |
| `CUDA Forward Compatibility mode ENABLED` at container start | The container CUDA is higher than the host driver CUDA | Normal behavior. Verify custom kernels separately in that environment |
