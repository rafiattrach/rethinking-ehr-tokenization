#!/bin/bash

# === Dynamic "No Value" Ablation Training Script ===
# Usage: bash ablation_runner_dynamic.sh task=<TASK_NAME> seed=<SEED> epochs=<EPOCHS>

# --- Parse Args ---
for arg in "$@"; do
  case $arg in
    task=*) TASK_NAME="${arg#*=}" ;;
    seed=*) SEED="${arg#*=}" ;;
    epochs=*) EPOCHS="${arg#*=}" ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# --- Validate Inputs ---
if [[ -z "$TASK_NAME" || -z "$SEED" || -z "$EPOCHS" ]]; then
  echo "Missing required arguments."
  echo "Usage: bash ablation_runner_dynamic.sh task=<TASK> seed=<SEED> epochs=<EPOCHS>"
  exit 1
fi

# --- Activate environment ---
source .rqvenv/bin/activate

# --- Set config ---
EXPERIMENT_NAME="triplet_mtr"
RUN_GROUP_NAME="no_value_ablation_multi"
SAFE_TASK_NAME="${TASK_NAME//\//_}"
ACCELERATOR="auto"
PRECISION="32"
STRATEGY="auto"
DEVICES=1
BATCH_SIZE=64
TOKEN_DIM=128
LOGGER="csv"

RUN_OUTPUT_DIR="results/${RUN_GROUP_NAME}/${EXPERIMENT_NAME}/${SAFE_TASK_NAME}/seed${SEED}_epochs${EPOCHS}"
mkdir -p "$RUN_OUTPUT_DIR"

# --- Run ---
echo "🔧 Training No-Value on $TASK_NAME | Seed=$SEED | Epochs=$EPOCHS"
echo ""

meds-torch-train \
  experiment=${EXPERIMENT_NAME} \
  paths.data_dir=triplet_tensors \
  paths.meds_cohort_dir=MEDS_cohort \
  paths.output_dir=${RUN_OUTPUT_DIR} \
  data.task_name=${TASK_NAME} \
  data.task_root_dir=MEDS_cohort/tasks \
  trainer.accelerator=${ACCELERATOR} \
  trainer.devices=${DEVICES} \
  trainer.precision=${PRECISION} \
  trainer.strategy=${STRATEGY} \
  logger=${LOGGER} \
  seed=${SEED} \
  ++model.token_dim=${TOKEN_DIM} \
  ++data.dataloader.batch_size=${BATCH_SIZE} \
  ++trainer.max_epochs=${EPOCHS} \
  ++data.dataloader.num_workers=6 \
  hydra.searchpath="[pkg://meds_torch.configs,meds-torch/MIMICIV_INDUCTIVE_EXPERIMENTS/configs/meds-torch-configs]"

exit $? 