# QEMU Boot

## Quick Start

QEMU 실행 파일과 guest 이미지가 준비된 환경에서, **host의 workspace root**에서 실행한다.
Single과 dual 중 하나만 선택한다.

**Host 터미널 1 — 부팅**

Single-app:

```bash
bash qemu_boot/run_qemu.sh 1 3
```

Dual-app:

```bash
QEMU_GEMMINI_POLICY=modulo bash qemu_boot/run_qemu.sh 2 5
```

**Host 터미널 2 — guest 부팅 완료 후 pinning (single/dual 공통)**

```bash
bash qemu_boot/pin.sh 8 2
```

각 thread의 `[OK]`를 확인한 뒤
[Power Modeling](../gemmini_experiments/power_modeling/README.md)의 안내에 따라 실험을 진행한다.
처음 구성하는 환경의 준비 사항과 SSH·공유 폴더 확인 방법은 아래에 설명한다.

## 상세 안내

Power modeling에 필요한 QEMU RISC-V guest 부팅과 host CPU pinning을 안내한다.
[run_qemu.sh](./run_qemu.sh)는 Gemmini 장치와 guest vCPU를 구성하고,
[pin.sh](./pin.sh)는 QEMU thread를 지정한 host CPU에 배치한다.

Guest 명령으로 표시한 경우를 제외하고, 모든 명령은 **host의 workspace root**에서 실행한다.

## 0. 부팅 준비

Workspace root에서 q-vela 저장소를 `./q-vela/`에 clone한다.

Ubuntu 24.04 기준 부팅용 패키지:

```bash
sudo apt update
sudo apt install -y opensbi u-boot-qemu qemu-system-misc
```

`run_qemu.sh`가 사용하는 파일은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `q-vela/qemu/build/qemu-system-riscv64` | Gemmini 장치를 지원하는 QEMU 실행 파일 |
| `ubuntu-24.04.3-preinstalled-server-riscv64.img` | Workspace root의 guest 디스크 이미지 |
| `/usr/lib/riscv64-linux-gnu/opensbi/generic/fw_jump.bin` | OpenSBI firmware |
| `/usr/lib/u-boot/qemu-riscv64_smode/uboot.elf` | U-Boot |
| `q-vela/tmp-cloudinit/id_ed25519` | Power runner가 사용하는 guest SSH 접속 키 |

패키지 설치만으로 Gemmini용 QEMU와 guest 이미지가 준비되는 것은 아니다.
`q-vela`와 이미지가 심볼릭 링크라면 실제 대상이 존재해야 한다.
Guest에는 SSH 서버와 해당 키로 접속할 수 있는 계정이 준비되어 있어야 한다.
현재 power runner의 기본 guest 계정은 `ubuntu`다.

## 1. 실험 구성에 맞춰 부팅

아래 두 구성 중 실행할 실험에 맞는 **하나만** 선택한다.
같은 디스크 이미지와 SSH 포트를 사용하는 QEMU를 동시에 실행하지 않는다.

### Single-app

Gemmini 1개와 vCPU 3개로 부팅한다.

```bash
bash qemu_boot/run_qemu.sh 1 3
```

### Dual-app

Gemmini 2개와 vCPU 5개로 부팅하고, executor별 Gemmini 배치를 위해 `modulo`를 사용한다.

```bash
QEMU_GEMMINI_POLICY=modulo bash qemu_boot/run_qemu.sh 2 5
```

일반 사용법은 `bash qemu_boot/run_qemu.sh [num_gemmini] [num_vcpu]`다.
인자는 **Gemmini 수 → vCPU 수** 순서이며, 생략 시 `2 2`다.
현재 스크립트는 Gemmini 1~4개, vCPU 1~12개를 허용한다.
Power 실험에서는 인자를 생략하지 말고 위 구성을 사용한다.

기본 binding policy는 `idle_first`이며, `modulo`에서는 guest `vCPU i`가
`Gemmini (i % num_gemmini)`를 사용한다. Dual의 executor vCPU 3과 4는 각각
Gemmini 1과 0에 연결된다.

QEMU는 현재 터미널에서 실행된다. 이후 host 명령은 다른 터미널에서 진행한다.
QEMU 로그는 `q-vela/qemu/qemu-gemmini.log`에 저장되며, 새로 부팅할 때 기존 로그는 지워진다.

## 2. Host CPU pinning

Guest가 부팅된 뒤 다른 host 터미널에서 실행한다. Single과 dual 모두 현재 기본값은 같다.

```bash
bash qemu_boot/pin.sh 8 2
```

첫 번째 인자는 vCPU의 시작 host CPU 번호, 두 번째는 Gemmini worker의 시작 번호다.
여기서 CPU 번호는 `taskset`이 사용하는 host 논리 CPU ID다.

| 대상 | 배치 규칙 | `8 2` 예시 |
|---|---|---|
| QEMU main·보조 thread | vCPU 시작 CPU | CPU 8 |
| Guest vCPU i | vCPU 시작 CPU + i | Single: 8~10, dual: 8~12 |
| Gemmini worker i | Gemmini 시작 CPU + i | Single: 2, dual: 2~3 |

출력에서 각 thread의 `[OK]`를 확인한다. 지정한 CPU가 host에서 사용 가능해야 하며,
CPU 배치를 바꿀 때는 power runner의 계측 대상도 함께 맞춰야 한다.
QEMU를 재부팅하면 thread ID가 달라지므로 pinning을 다시 실행한다.

별도 QEMU 로그를 지정할 때는 다음 형식을 사용한다.

```bash
bash qemu_boot/pin.sh q-vela/qemu/qemu-gemmini.log 8 2
```

## 3. 공유 폴더와 SSH 확인

Host의 `gemmini_experiments/`를 guest의 `/mnt/hostshare`에서 사용할 수 있어야 한다.
부팅 스크립트는 9p 공유 장치를 제공하며, guest 내 mount는 별도로 필요하다.

Guest 터미널에서 확인한다.

```bash
mountpoint /mnt/hostshare
ls /mnt/hostshare/applications /mnt/hostshare/power_modeling
```

아직 mount되지 않았다면 guest에서 실행한다.

```bash
sudo mkdir -p /mnt/hostshare
sudo mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/hostshare
```

Host의 TCP 2200 포트가 guest의 SSH 22 포트로 연결된다.
Host 터미널에서 power runner와 같은 키로 접속을 확인한다.

```bash
ssh -i q-vela/tmp-cloudinit/id_ed25519 -p 2200 ubuntu@127.0.0.1
```

공유 폴더에서 application 파일을 읽고 실험 파일을 기록할 수 있어야 한다.
부팅·pinning·SSH·공유 폴더 준비가 끝나면 해당 실험으로 이동한다.

- [Power Modeling](../gemmini_experiments/power_modeling/README.md)
