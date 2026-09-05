# ChipletFM

ChipletFM is a flow-matching based chiplet placement model with thermal-aware evaluation and HotSpot integration.

This repo is organized around the `layout` dataset. The main entry points are:

- `diffusion/train_graph.py` for training and fine-tuning
- `diffusion/eval_thermal_guided.py` for evaluation
- `scripts/run_case1_10.sh` for the end-to-end Case1-10 benchmark pipeline

## Requirements

Use the `chipdiffusion` conda environment or install the pinned dependencies from:

```bash
pip install -r ChipletFM/requirements.txt
```

External system dependencies are not installed by pip:

- IBM CPLEX Python bindings
- TAP-2.5D
- HotSpot

For the benchmark scripts, set these if they are not already in your shell:

```bash
export CPLEX_STUDIO_DIR=/opt/ibm/ILOG/CPLEX_Studio221
export TAP25D_ROOT=/mnt/d/WORK/NEW/TAP-2.5D
```

## Dataset

The default dataset name is `layout`.

Expected directories:

- `datasets/graph/layout`
- `datasets/graph/layout_test`

The benchmark Case1-10 data lives under:

- `benckmark/atplace_case1_10`

## Training

Train from scratch with the `layout` config:

```bash
cd ChipletFM
python diffusion/train_graph.py --config-name config_graph_fm
```

Common overrides:

```bash
python diffusion/train_graph.py --config-name config_graph_fm \
  task=layout \
  method=layout-train \
  seed=61 \
  train_steps=3000000 \
  batch_size=64 \
  val_batch_size=64
```

Training writes checkpoints and logs under `logs/diffusion_debug` by default. The best checkpoint is saved as:

```bash
logs/diffusion_debug/<task>/<method>/seed_<seed>/best.ckpt
```

## Fine-tuning

Fine-tuning uses the same training entry point, but switches `mode` to `finetune` and points `from_checkpoint` at a pre-trained model:

```bash
python diffusion/train_graph.py --config-name config_graph_fm \
  mode=finetune \
  task=layout \
  method=layout-finetune \
  from_checkpoint=checkpoints/model/best.ckpt \
  seed=61
```

Behavior in `finetune` mode:

- if an existing `latest.ckpt` is present in the run directory, training resumes from it
- otherwise the model weights are initialized from `from_checkpoint`
- optimizer state is resumed only if a matching checkpoint exists locally

## Evaluation

The main evaluation entry point is:

```bash
python diffusion/eval_thermal_guided.py --config-name config_eval_fm
```

Typical overrides:

```bash
python diffusion/eval_thermal_guided.py --config-name config_eval_fm \
  task=layout \
  method=layout-eval \
  from_checkpoint=checkpoints/model/best.ckpt \
  thermal.ckpt=checkpoints/thermal/thermal_ep0175.pth \
  eval_samples=0 \
  num_output_samples=12
```

This produces model outputs, thermal metrics, and per-run artifacts under:

```bash
logs/output/<task>/<method>/seed_<seed>/
```

## Case1-10 benchmark

To run the full Case1-10 pipeline:

```bash
cd ChipletFM/scripts
bash run_case1_10.sh
```

For a quick smoke test with one sample:

```bash
CONDA_SH=/home/user/miniconda3/etc/profile.d/conda.sh \
CONDA_ENV=chipdiffusion \
CPLEX_STUDIO_DIR=/opt/ibm/ILOG/CPLEX_Studio221 \
TAP25D_ROOT=/mnt/d/WORK/NEW/TAP-2.5D \
RUN_CASE1_10_SINGLE=1 \
METHOD=smoke-hotspot-fast \
SEED=13012 \
EVAL_EXTRA_ARGS='num_output_samples=1 model.max_diffusion_steps=2 legalization.grad_descent_steps=1 thermal.guidance_steps=1' \
bash scripts/run_case1_10.sh
```

HotSpot outputs are written under:

```bash
logs/output/<task>/<method>/seed_<seed>/hotspot/
```

And the summary files are:

- `hotspot_temperatures.csv`
- `hotspot_temperatures.json`

## Notes

- `wandb` is disabled by default in the open-source configuration.
- Large datasets, checkpoints, logs, and generated outputs are ignored by git.
- If you only want model inference without HotSpot, set `SKIP_HOTSPOT=1`.
