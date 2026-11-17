# Applications

This guide describes how to prepare the models and input data for the applications needed for QEMU power modeling experiments,
and how to build the Gemmini power binaries. Run all commands from the **workspace root on the host**.
QEMU does not need to be booted for building.

## Supported Applications

| Application | Purpose | Guest source/binary location |
|---|---|---|
| DistilBERT | SST-2 text classification | [distilbert/sst2_pipeline/guest](./distilbert/sst2_pipeline/guest/) |
| DistilRoBERTa | SST-2 text classification | [distilroberta/sst2_pipeline/guest](./distilroberta/sst2_pipeline/guest/) |
| DistilGPT2 | SST-2 text classification | [distilgpt2/sst2_pipeline/guest](./distilgpt2/sst2_pipeline/guest/) |
| PolyBench GEMM | Matrix multiplication benchmark | [polybench_gemm/guest](./polybench_gemm/guest/) |

## 0. Prerequisites

- `make`, `riscv64-linux-gnu-gcc`, and a static C library for RISC-V Linux
- The Gemmini headers in `q-vela/qemu/test-code/gemmini_test_code/include/`
- When generating Transformer model data, a host Python environment with Python 3.10 or later and `torch`, `numpy`,
  `transformers`, `datasets`, and `peft` installed
- A network connection for downloading if the models, tokenizers, or SST-2 data are not in the cache

If the C build tools are missing on an Ubuntu-based host, install them as follows.

```bash
sudo apt install make gcc-riscv64-linux-gnu libc6-dev-riscv64-cross
```

If `q-vela` is a symbolic link, it must point to the actual path.
The Makefile uses the Gemmini header path above by default; to use a different location,
pass `GEMMINI_INCLUDE=/absolute/path/to/include` as a make argument.

## 1. Preparing Transformer Models and Input Data

In addition to the executable binary, Transformers require the data in `<app>/sst2_pipeline/guest/generated/`.
If the data used for existing experiments is already prepared, skip this step and
proceed to building the binaries. PolyBench does not require separate model data.

For the first preparation, run the existing data generation pipeline for the three applications
in order with the command below. `03_export_c_model.py` currently reads the outputs of `00`, `01`, and `02`,
so do not skip the intermediate steps.

```bash
(
    set -e
    for app in distilbert distilroberta distilgpt2; do
        pipeline="gemmini_experiments/applications/${app}/sst2_pipeline"
        python3 "${pipeline}/00_fp32_baseline.py" --allow-download
        python3 "${pipeline}/01_quantize_linear_int8.py" --allow-download
        python3 "${pipeline}/02_eval_host_int8.py" --allow-download
        python3 "${pipeline}/03_export_c_model.py" --allow-download
    done
)
```

To prepare only one application, leave only its name in the `for app in ...` list.
The command above uses the models defined in the scripts and the default input length of 128.
It may take a while, as it includes model downloads and host computation.

Intermediate outputs are generated in `<app>/sst2_pipeline/outputs/`, and guest data in the following location.

```text
<app>/sst2_pipeline/guest/generated/
├── export_manifest.json
├── model_bin/
├── eval_bin/
└── debug_bin/
```

Re-running updates the existing outputs, and the export step deletes and regenerates `model_bin/`, `eval_bin/`,
and `debug_bin/`. If there is data you want to keep, back it up separately first.
If you change the model structure or input length, you must also check the workload cycle profile of the power model.

## 2. Building All Gemmini Power Binaries

Build only the four binaries used by the current power runner.

```bash
(
    set -e
    for app in distilbert distilroberta distilgpt2; do
        make -B -C "gemmini_experiments/applications/${app}/sst2_pipeline/guest" \
            "${app}_gemmini_power_infer.riscv"
    done

    make -B -C gemmini_experiments/applications/polybench_gemm/guest \
        gemmini-power DATASET=MEDIUM_DATASET
)
```

`-B` rebuilds even if the binaries already exist. In particular, a PolyBench dataset change is
not detected from file modification times alone, so a forced build is used.
To match the current Gemmini workload cycle profile, PolyBench is built with `MEDIUM_DATASET`.

The generated files are as follows.

```text
distilbert/sst2_pipeline/guest/distilbert_gemmini_power_infer.riscv
distilroberta/sst2_pipeline/guest/distilroberta_gemmini_power_infer.riscv
distilgpt2/sst2_pipeline/guest/distilgpt2_gemmini_power_infer.riscv
polybench_gemm/guest/polybench_gemm_gemmini_power.riscv
```

Individual builds use the same targets. For example, to build only DistilBERT:

```bash
make -B -C gemmini_experiments/applications/distilbert/sst2_pipeline/guest \
    distilbert_gemmini_power_infer.riscv
```

## 3. Running Power Experiments

The prepared binaries are run on the RISC-V guest. Follow the documents below for QEMU boot,
measurement, and post-processing.

- [QEMU Boot and Pinning](../../qemu_boot/README.md)
- [Power Modeling](../power_modeling/README.md)

## Auxiliary Tool: Tiler

[gemmini_ws_tiler.c](./gemmini_tiler/gemmini_ws_tiler.c) and
[transformer_gemmini_tiler.c](./transformer_gemmini_tiler.c) are **auxiliary tools for
cycle experiment analysis**. They analyze the tile decomposition of GEMM sizes and the per-application Gemmini API and `LOOP_WS` call composition,
and are not a required step for building the power binaries or running experiments.
