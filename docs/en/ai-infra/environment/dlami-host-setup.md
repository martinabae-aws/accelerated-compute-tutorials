---
title: DLAMI Host Setup and Validation
description: How to build the host layer of a large-scale GPU training cluster on top of DLAMI — how to choose a DLAMI family, and how to validate pre-installed components such as GPU, driver, EFA, and GDRCopy (a common procedure independent of orchestrator and storage)
---

# DLAMI Host Setup and Validation

This page covers the **host layer** of a GPU training cluster. It explains which AMI to use to provision the operating system, NVIDIA driver, and networking (EFA) stack that every GPU node must have in common, and how to validate that this configuration is correct after boot. The scheduler, container runtime, and training runtime all sit on top of this host layer (common whether the orchestrator is Slurm, EKS, or ParallelCluster), so a cleanly validated host here lets you quickly isolate problems in the upper layers.

There is a single core message: **do not hand-assemble the low-level components required for GPU, EFA, and high-speed storage; instead, use an AMI that ships them bundled together with versions already aligned.** And since the most common mistake that breaks that alignment is a kernel upgrade, this page repeatedly emphasizes the principle of "do not upgrade the kernel."

## DLAMI families and how to choose

AWS Deep Learning AMIs (DLAMIs) are not a single image but **several families**. The pre-installed software and the way you operate differ by family, so decide which family fits your workload before you start building the host.

### Family breakdown

DLAMIs differ mainly in **what comes pre-installed**.

- **Base family** — includes only the lower-level stack (NVIDIA driver, CUDA, EFA, container toolkit, etc.) and does **not** include a training framework (such as PyTorch). It assumes the framework is run as a container.
- **Framework family** — a specific framework (PyTorch, etc.) comes **pre-installed** along with the driver and CUDA. Suited to running training directly on the host without containers.
- **Multi-framework / other families** — variants that bundle multiple frameworks or carry only a minimal configuration, depending on the use case.

> DLAMI families, names, and contents are updated periodically by AWS, so check the latest list in the [DLAMI Developer Guide](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html) when making your choice.

### Decision points

Which DLAMI fits the large-scale GPU training **host layer** is judged by the following.

- **Do you run the training framework as a container, or install it directly on the host?** — if you containerize it, a framework-free Base family is lighter and more reproducible. If you run it directly on the host, a Framework family is more convenient.
- **Must every node share an identical environment?** — large-scale distributed training requires all nodes to have the same software environment. Pinning the environment in a container image lets you keep the host light.
- **Are the EFA, GDRCopy, and high-speed-storage kernel modules aligned with the driver and kernel?** — whatever family you use, the key is whether it meets the "host requirements" below.
- **Will you bake and reuse your own operational AMI?** — if you layer the scheduler and runtime on top of the base and bake an operational AMI, a family whose kernel and kernel modules are held (hold) is easier to maintain.

### Host requirements (common, family-independent)

Whatever DLAMI (or custom AMI) you use, a GPU training host needs a set of components with strong **mutual version dependencies**, present **in alignment with the running kernel**.

- The **NVIDIA driver** and the CUDA runtime on top of it
- **Fabric Manager**, which binds multiple GPUs into a single fabric (NVLink/NVSwitch topology management)
- **EFA kernel driver, libfabric, and aws-ofi-nccl**, which handle high-speed inter-node communication
- **GDRCopy**, which enables direct GPU memory transfers
- **NVIDIA Container Toolkit**, which exposes GPUs to containers
- (when using FSx for Lustre) **Lustre client kernel modules** for mounting shared storage

In particular, the **EFA kernel driver** (and, if you use FSx for Lustre, the Lustre client) is a kernel module, so if the kernel changes, those modules may no longer match and networking communication or storage mounts can break. Installing all of this by hand to align versions is poorly reproducible and error-prone. That is why it is safer to **use an AMI that ships these requirements already met and bundled together**.

!!! info "Scope of this guide"
    From here on, this guide covers host setup and validation for the case where you chose the **Base OSS family (Deep Learning Base OSS Nvidia Driver GPU AMI)**. This family already meets the requirements above and suits large-scale distributed training where the training framework is run as a container. Guides for other DLAMI families will be added later.

## Base OSS family — composition and characteristics

The AWS-provided **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)** ships the components above **already bundled with versions aligned to one another**. Because it is in the "Base" family, it does not include a specific framework (such as PyTorch); it carries only the lower-level stack of driver, EFA, toolkit, and so on, which suits a configuration where the training framework is run as a container. The recommended flow is to use this AMI as the baseline and build an operational AMI on top of it by adding the upper layers such as the scheduler and container runtime.

!!! note "Validated environment (as of 2026-10)"
    AMI IDs differ by region, so look them up via SSM Public Parameter.

    ```bash
    aws ssm get-parameter --region <region> \
      --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
      --query Parameter.Value --output text
    ```

    The build used for validation was `20261002`, and it came with the following configuration pre-installed.

    | Component | Version |
    |---|---|
    | Kernel | 7.0.0-1014-aws |
    | NVIDIA driver / CUDA (driver-level) | 595.91.07 / 13.2 |
    | Fabric Manager | active |
    | EFA driver / EFA installer | 3.3.0 / 1.50.0 (libfabric 2.6.0, aws-ofi-nccl 1.21.1, Open MPI 4.1.7·5.0.9, PMIx 4.2.8) |
    | GDRCopy | `gdrdrv` loaded, `/dev/gdrdrv` |
    | NVIDIA Container Toolkit / Docker | 1.20.1 / 29.8.2 |
    | Lustre client | 2.15.6 (module for the current kernel) |
    | nccl-tests | `/usr/local/cuda-XX/efa/test-cuda-XX/` (CUDA 12.8·12.9·13.0·13.2) |
    | Local NVMe | 8 drives bundled with LVM and auto-mounted at `/opt/dlami/nvme` (about 28 TB) |

    The versions above are a snapshot at a specific point in time. When the AMI build is updated, each component version changes as well, so treat the values you look up in your actual environment as authoritative.

> 📎 [DLAMI Developer Guide](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)

## Do not upgrade the kernel (the most important principle)

In DLAMI, the **kernel and kernel-module packages are held (hold)**. This is by design.

- The EFA kernel driver is a **kernel module bound to the running kernel version**. (If you use FSx for Lustre, the Lustre client is one as well.)
- If you upgrade the kernel, there are no modules matching the new kernel, so **EFA communication may break** (and, if you use FSx for Lustre as shared storage, its mounts may fail too).
- Therefore, do not upgrade the kernel with `apt upgrade` or similar, and keep the hold state in place.

This principle applies even when you build a new operational AMI. **Keep the kernel version pinned when building an operational AMI, too.** If a security patch is needed, instead of upgrading the kernel, it is safer to replace the base with a new DLAMI build (a build in which all modules are re-aligned).

!!! warning "Typical symptoms triggered by a kernel change"
    - EFA device not recognized or communication initialization failure — EFA kernel driver mismatch
    - (when using FSx for Lustre) mount failure — the Lustre kernel module does not match the new kernel

    If a problem occurs, first check with `uname -r` whether the running kernel has changed from the AMI's default kernel.

## CUDA Forward Compatibility — why container CUDA can be higher than the host

Because the training framework runs as a container, **the CUDA version inside the container can be higher than the CUDA version of the host driver.** What makes this work is **CUDA Forward Compatibility**.

- Through the Forward Compatibility package, the NVIDIA driver supports CUDA runtimes newer than itself.
- That is, you can keep the host kernel and driver fixed as DLAMI pins them, and bring only the container up to a more recent CUDA, and GPU compute, NCCL, EFA communication, and training will all work correctly.
- If `CUDA Forward Compatibility mode ENABLED` appears in the logs at container startup, it means the container is running in this mode; this is not an error but a normal state.

Thanks to this mechanism, a layer separation of **"keep the kernel and driver stably pinned, and freely update the framework via containers"** is achieved. This is also why you do not need to upgrade the kernel.

!!! note "Validated environment (as of 2026-10)"
    - Host driver CUDA: 13.2
    - Container (NGC `nvcr.io/nvidia/pytorch:26.08-py3`) CUDA: 13.4.1
    - Result: Confirmed that GPU compute and NCCL/EFA communication work correctly in Forward Compatibility mode.

    If your training code includes custom CUDA kernels, we recommend separately verifying that they build and run under the Forward Compatibility environment. Custom kernels compiled with a CUDA toolkit higher than the driver CUDA should be checked against the supported range of this mode.

## Host validation

After booting the instance, verify **on each node** that all pre-installed components are in the correct state with the commands below. Run this validation before bringing up Slurm or containers, so you can first determine whether a problem is in the host layer or in an upper layer.

```bash
nvidia-smi --query-gpu=index,name,driver_version --format=csv   # GPU recognition and driver
systemctl is-active nvidia-fabricmanager                        # Fabric Manager status
modinfo efa | grep -E "^version"                                # EFA kernel driver version
cat /opt/amazon/efa_installed_packages | head -3                # EFA installer configuration
ls /sys/class/infiniband/ | grep -c rdmap                       # EFA device count
fi_info -p efa -t FI_EP_RDM | grep domain | sort -u             # EFA domains seen by libfabric
lsmod | grep gdrdrv; ls -l /dev/gdrdrv                           # GDRCopy
ulimit -l                                                       # memlock (pinned memory limit)
df -h /opt/dlami/nvme                                            # local NVMe mount
```

!!! note "When using FSx for Lustre as shared storage"
    If you use FSx for Lustre as shared storage, validate the Lustre client as well. (If you use other storage such as S3 or FSx for OpenZFS, this check is unnecessary.)

    ```bash
    lfs --version   # check Lustre client installation and version
    ```

    To use **GPUDirect Storage (GDS)**, where the GPU reads data directly from storage, additional checks are required. See the [GDS (GPUDirect Storage)](gds-gpudirect-storage.md) guide.

### What each command checks

- **`nvidia-smi`** — whether all GPUs are recognized and the driver is loaded. The GPU count and model matching the instance type should be shown.
- **`nvidia-fabricmanager`** — the service that manages the NVLink/NVSwitch fabric. If it is not `active`, multi-GPU collective communication (the NVLink path) will not be established.
- **`modinfo efa`** — whether the EFA kernel driver is installed and its version. This is the foundation of high-speed inter-node communication.
- **`efa_installed_packages`** — check at a glance the versions of libfabric, aws-ofi-nccl, and others that the EFA installer laid down together.
- **the number of `rdmap*` entries under `/sys/class/infiniband/`** and **the number of domains from `fi_info`** — check the number of actually exposed EFA devices at both the OS level and the libfabric level. The two values should match the number of EFA devices on the instance.
- **`gdrdrv` / `/dev/gdrdrv`** — whether the GDRCopy kernel module for the GPUDirect RDMA path is loaded and the device node exists.
- **`ulimit -l` (memlock)** — collective communication libraries hold large amounts of pinned memory, so this should be `unlimited`. If it is limited, communication initialization or performance problems arise.
- **`/opt/dlami/nvme`** — whether the local NVMe is bundled with LVM and auto-mounted. It is used for container image cache and scratch.
- **(when using FSx for Lustre) `lfs --version`** — whether the Lustre client is installed so that shared storage can be mounted.

### Expected results

| Item | Expected result |
|---|---|
| GPU | All GPUs matching the instance type recognized |
| Fabric Manager | `active` |
| EFA driver | Version shown by `modinfo efa` |
| EFA device count | `rdmap*` count = `fi_info` domain count = number of EFAs on the instance |
| GDRCopy | `gdrdrv` loaded, `/dev/gdrdrv` present |
| memlock | `unlimited` |
| Local NVMe | `/opt/dlami/nvme` mounted |

!!! note "Validated environment (as of 2026-10)"
    These are the baseline values measured on P6-B200.48xlarge (8 GPUs, 8 EFAs per instance).

    | Item | Expected result |
    |---|---|
    | GPU | NVIDIA B200 × 8 |
    | Fabric Manager | `active` |
    | EFA driver | 3.3.0 or higher |
    | EFA device count | 8 `rdmap*`, 8 `fi_info` domains |
    | GDRCopy | `gdrdrv` loaded, `/dev/gdrdrv` present |
    | memlock | `unlimited` |

### When ConnectX-7 devices also appear

On some instance types, besides EFA (`rdmap*`), **ConnectX-7 devices such as `ibp115s0f0` and `ibp116s0f0` may also appear under `/sys/class/infiniband/` with their links DOWN.** This is normal and has no effect on EFA communication.

- NCCL selects EFA (the communication log shows `Using network Libfabric`, `provider is efa`, and `found N nics`).
- Therefore no separate action is needed. **Do not force the IB path with `NCCL_NET=IB` or similar** — doing so forces the link-DOWN ConnectX-7 into use and communication initialization fails.

### Single-node NCCL check (NVLink only)

First verify, within a node, that GPU, driver, and NCCL are healthy at the host level. (This is a host-only check performed before installing the scheduler.) A single-node test does not go through EFA and uses only intra-node NVLink, so it is useful for separating inter-node networking problems from GPU/driver problems.

```bash
TESTDIR=/usr/local/cuda-13.2/efa/test-cuda-13.2
/opt/amazon/openmpi/bin/mpirun -n 8 --bind-to none \
  -x LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib:/usr/local/cuda-13.2/lib64:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH \
  -x NCCL_DEBUG=INFO \
  $TESTDIR/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 50
```

Check the following in the result log.

- `Out of bounds values : 0 OK` — computational correctness. If it is not 0, it is a failure regardless of performance.
- busbw value — whether the GPUs and NVLink deliver normal bandwidth. The single-node value becomes the baseline for subsequent multi-node (EFA) tests.

!!! note "Validated environment (as of 2026-10)"
    - nccl-tests path: `/usr/local/cuda-13.2/efa/test-cuda-13.2`
    - P6-B200 single-node all-reduce (1 GiB) reference value: **busbw 711 GB/s**

    The CUDA version portion of the nccl-tests path (`cuda-13.2`) changes to match the CUDA version included in the AMI. Check the paths under `/usr/local/` in your actual environment and use them. The performance figure above is a reference value measured at a specific point in time and configuration, and is not a guaranteed value.

## Summary

- The host layer is based on the **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)**. The driver, Fabric Manager, EFA, GDRCopy, and Container Toolkit are bundled with versions aligned to one another (plus the Lustre client if you use FSx for Lustre).
- **Do not upgrade the kernel.** The kernel and its kernel modules (the EFA driver, and the Lustre client when using FSx for Lustre) are held, and a kernel change breaks this alignment. Keep the kernel pinned even when building an operational AMI.
- Even when the container CUDA is higher than the host driver CUDA, it works correctly via **CUDA Forward Compatibility**. This is what enables "keep the kernel pinned, update the framework via containers."
- On each node, **always validate** `nvidia-smi`, Fabric Manager, EFA (driver, device count, libfabric), GDRCopy, and memlock **before** moving on to the upper layers. (Also check the Lustre client if you use FSx for Lustre.)
