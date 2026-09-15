#!/usr/bin/env bash
set -euo pipefail

# Flow-matching training on the 20k Pareto-optimised placement set.
#
# Dataset: Dataset/dataset/placement_dataset/placement_dataset_opt_pareto (20,000 systems),
# split by Dataset/splits/placement_pareto_20k into 16000 train / 2000 val / 2000 test.
#
# Unlike placement_opt_20k, whose geometry was produced by a connectivity-aware
# greedy/FD placer that is blind to thermal, this set is the per-system Pareto
# front over (peak temperature, CPLEX wirelength, bbox area) as scored by the two
# frozen surrogates, with one front member written per system.  The source is
# placement_dataset_tw chunks 5-8 (ids 20001-40000, renumbered to 1-20000), so
# the systems are disjoint from placement_opt_20k (chunks 1-4) -- no leakage, and
# the retrained-vs-baseline comparison is clean.
#
# NOTE: this run's auxiliary losses are computed against the *same* frozen
# surrogates the dataset was optimised with, so a lower aux loss here is NOT
# evidence that the dataset is better.  Only the real-physics evaluation
# (eval_layout.py) counts.
#
# Differences from train_flow_first_300k_wsl.sh:
#   * repo_root defaults to the directory holding this script (works on the WSL
#     mount and on a plain Linux checkout) instead of a hard-coded /mnt/d path.
#   * task is overridden to placement_pareto_20k, so the dataset, the log directory
#     and the monitor path all switch together.
#   * train_steps defaults lower.  GraphDataLoader.get_batch draws a single
#     layout per step and expands it across the batch (see the "# TODO support
#     larger batch sizes" note in utils.py), so one step covers one unique
#     layout regardless of batch_size.  An epoch over the 16k train split is
#     therefore 16000 steps, not 16000/batch_size.  The 300k reference ran 3M
#     steps over a 240k train split, i.e. ~12.5 epochs; 200k steps here is the
#     same ~12.5 epochs, and 16000 steps is one epoch.
#   * PYTHONPATH picks up _env_compat/sitecustomize.py, which restores the
#     NumPy<2 aliases that the pinned wandb release needs at import time.
#   * Checkpoints land in LayoutGenModel/checkpoints/model_pareto, not logs/.  The
#     trainer always inserts model/<task>/<method>/seed_<seed> under log_root, so
#     the run is symlinked into the flat model_pareto/placement_pareto_20k/seed_61
#     while it trains and moved there for real once it finishes.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${FLOW_TAP_ROOT:-$script_dir}"

if [[ -n "${FLOW_TAP_PYTHON:-}" ]]; then
  python_executable="$FLOW_TAP_PYTHON"
elif [[ -x /root/anaconda3/envs/chipdiffusion/bin/python ]]; then
  python_executable=/root/anaconda3/envs/chipdiffusion/bin/python
elif [[ -x /home/user/miniconda3/envs/chipdiffusion/bin/python ]]; then
  python_executable=/home/user/miniconda3/envs/chipdiffusion/bin/python
else
  echo "No chipdiffusion interpreter found; set FLOW_TAP_PYTHON." >&2
  exit 1
fi

train_steps=200000
batch_size=8
val_batch_size=8
# Mega-batching: one training step collates `layouts_per_step` distinct layouts
# into a single mega-graph and expands it across `noise_per_layout` independent
# (t, noise) samples, so the model does one batched forward over
# layouts_per_step x noise_per_layout samples instead of one layout repeated.
# 16 x 8 = 128 effective batch, verified bit-equivalent (up to fp16 autocast
# rounding) to running each layout alone.
layouts_per_step=16
noise_per_layout=8
monitor_every=100
print_every=500
method="placement-pareto-20k-thermal-wirelength-legality"

# Auxiliary-loss weights.  The config defaults (thermal 0.01 / wirelength 0.02)
# left both terms effectively unlearned: over the 16k-step run their training
# losses were flat (thermal 0.5315 -> 0.5347, wirelength 12.47 -> 12.42) while
# flow_loss fell 33%.  thermal contributed only 0.66% of the total loss, so it
# is raised an order of magnitude.  wirelength already contributed 31% of the
# total loss at 0.02, so it is only nudged; the reason it did not move is the
# t-weighting, not the weight (see aux_t_reweight below).
thermal_weight=0.1
wirelength_weight=0.05

# train_graph_thermal.py:392 documents that d L_aux / d v = -t * d L_aux / d x_hat,
# so with t ~ U(1e-4, 1) roughly 43x more auxiliary gradient reaches the model at
# t~1 (where x_hat is noise, and the thermal/wirelength of E[x|z] is a constant
# the model cannot lower) than at t~0 (where x_hat is the actual layout).  That
# flag is default-OFF and is absent from config_graph_fm.yaml, i.e. it was off in
# the 16k run.  Reweighting by 1/clamp(t) makes every t contribute equally.
aux_t_reweight=true

# Aux weights ramp linearly from 0 to full over this many steps.  The 16k run
# used the config default of 10000, so full strength was only reached for the
# last 6000 steps.  Shorter here so the aux terms have traction well before the
# run ends.
aux_warmup_steps=2000

# Start a new run instead of resuming whatever is already at
# placement_pareto_20k/seed_61.  Needed when the point of the run is to compare
# against the previous one.
fresh=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-steps) train_steps="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --val-batch-size) val_batch_size="$2"; shift 2 ;;
    --layouts-per-step) layouts_per_step="$2"; shift 2 ;;
    --noise-per-layout) noise_per_layout="$2"; shift 2 ;;
    --monitor-every) monitor_every="$2"; shift 2 ;;
    --print-every) print_every="$2"; shift 2 ;;
    --method) method="$2"; shift 2 ;;
    --thermal-weight) thermal_weight="$2"; shift 2 ;;
    --wirelength-weight) wirelength_weight="$2"; shift 2 ;;
    --aux-warmup-steps) aux_warmup_steps="$2"; shift 2 ;;
    --no-aux-t-reweight) aux_t_reweight=false; shift ;;
    --aux-t-reweight) aux_t_reweight=true; shift ;;
    --fresh) fresh=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

layout_root="$repo_root/LayoutGenModel"
train_entry="$layout_root/diffusion/train_graph_thermal.py"
compat_dir="$layout_root/diffusion/_env_compat"
required_paths=(
  "$python_executable"
  "$train_entry"
  "$compat_dir/sitecustomize.py"
  "$repo_root/thermalmodel/checkpoints/gnnhrnet_pwin/best.pth"
  "$repo_root/wirelengthmodel/checkpoint/best_wlmodel_total_60k.pt"
  "$repo_root/Dataset/splits/placement_pareto_20k/manifest.yaml"
  "$repo_root/Dataset/dataset/placement_dataset/placement_dataset_opt_pareto/chiplet_dataset_1.json"
)
for required_path in "${required_paths[@]}"; do
  [[ -e "$required_path" ]] || { echo "Required training input is missing: $required_path" >&2; exit 1; }
done

log_root="$layout_root/checkpoints/model_pareto"
run_dir="$log_root/model/placement_pareto_20k/$method/seed_61"
if [[ "$fresh" == true ]]; then
  # Give a fresh run its own directory.  Without this the resume block below
  # would find the *previous* run sitting at placement_pareto_20k/seed_61, move it
  # into this method's run_dir, and continue from its checkpoint -- silently
  # turning a new experiment into a resume of the old one.
  final_dir="$log_root/placement_pareto_20k/$method/seed_61"
else
  final_dir="$log_root/placement_pareto_20k/seed_61"
fi

# A previous run of this method was flattened here.  Move it back to the path
# the trainer expects so it resumes from latest.ckpt instead of starting over.
if [[ "$fresh" != true && -d "$final_dir" && ! -L "$final_dir" ]]; then
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite an existing run: $run_dir" >&2
    echo "Move it aside or pass --fresh." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$run_dir")"
  mv "$final_dir" "$run_dir"
  echo "Resuming from existing run: $run_dir"
fi

mkdir -p "$(dirname "$final_dir")"
# Point the final path at the in-progress run so the monitor and checkpoints are
# reachable there from the first step; replaced by the real directory afterwards.
ln -sfn "$run_dir" "$final_dir"

echo "Repo root: $repo_root"
echo "WSL Python: $python_executable"
echo "Dataset: Dataset/dataset/placement_dataset/placement_dataset_opt_pareto (146080 train / 18262 val)"
train_layouts=146080
echo "Training steps: $train_steps"
echo "  Draws $train_steps distinct layouts = $(awk -v s="$train_steps" -v n="$train_layouts" 'BEGIN{printf "%.1f", s/n}') epochs over the $train_layouts-layout train split."
echo "  Each step collates $layouts_per_step distinct layouts into one mega-graph and"
echo "  expands it across $noise_per_layout noise samples => $((layouts_per_step * noise_per_layout)) forward samples/step."
echo "Checkpoints: $final_dir"
echo "Live monitor: $final_dir/training_monitor.png"
echo "The PNG is replaced every $monitor_every training steps."
echo "Aux losses: thermal=$thermal_weight wirelength=$wirelength_weight"
echo "  Weights ramp over $aux_warmup_steps steps, then hold at full."
echo "  Aux t reweighting (1/t): $aux_t_reweight"

cd "$layout_root"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONPATH="$compat_dir${PYTHONPATH:+:$PYTHONPATH}"
set +e
"$python_executable" "$train_entry" \
  --config-name config_graph_fm \
  "task=placement_pareto_20k" \
  "method=$method" \
  "train_steps=$train_steps" \
  "batch_size=$batch_size" \
  "val_batch_size=$val_batch_size" \
  "+layouts_per_step=$layouts_per_step" \
  "+noise_per_layout=$noise_per_layout" \
  "monitor.every=$monitor_every" \
  "print_every=$print_every" \
  "thermal.train_weight=$thermal_weight" \
  "thermal.warmup_steps=$aux_warmup_steps" \
  "wirelength.train_weight=$wirelength_weight" \
  "wirelength.warmup_steps=$aux_warmup_steps" \
  "+flow.aux_t_reweight=$aux_t_reweight" \
  "+log_root=$log_root"
status=$?
set -e

# Flatten model/<task>/<method>/seed_<seed> down to <task>/seed_<seed>.
if [[ -d "$run_dir" ]]; then
  rm -f "$final_dir"
  mv "$run_dir" "$final_dir"
  find "$log_root/model" -type d -empty -delete 2>/dev/null || true
  echo "Run directory: $final_dir"
fi

exit $status
