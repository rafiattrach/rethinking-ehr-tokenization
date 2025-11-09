#!/bin/bash

# === Dynamic RQ1 LeTE Training Script ===
# Usage:
# bash run_rq1_lete_dynamic.sh task=<TASK_NAME> lete_variant=<balanced|fourier_heavy|spline_heavy> seed=<SEED> epochs=<EPOCHS>

# --- Parse Args ---
for arg in "$@"; do
  case $arg in
    task=*) TASK_NAME="${arg#*=}" ;;
    lete_variant=*) LETE_VARIANT="${arg#*=}" ;;
    seed=*) SEED="${arg#*=}" ;;
    epochs=*) EPOCHS="${arg#*=}" ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# --- Validate Inputs ---
if [[ -z "$TASK_NAME" || -z "$LETE_VARIANT" || -z "$SEED" || -z "$EPOCHS" ]]; then
  echo "Missing required arguments."
  echo "Usage: bash run_rq1_lete_dynamic.sh task=<TASK> lete_variant=<balanced|fourier_heavy|spline_heavy> seed=<SEED> epochs=<EPOCHS>"
  exit 1
fi

# --- Set LeTE Parameters based on variant ---
case $LETE_VARIANT in
  "balanced")
    P_FOURIER=0.5
    FOURIER_K=5
    SPLINE_HIDDEN_DIM=16
    ;;
  "fourier_heavy")
    P_FOURIER=0.8
    FOURIER_K=10
    SPLINE_HIDDEN_DIM=16
    ;;
  "spline_heavy")
    P_FOURIER=0.2
    FOURIER_K=3
    SPLINE_HIDDEN_DIM=32
    ;;
  *)
    echo "Invalid LeTE variant: $LETE_VARIANT"
    echo "Valid options: balanced, fourier_heavy, spline_heavy"
    exit 1
    ;;
esac

# --- Activate environment ---
source .rqvenv/bin/activate

# --- Set config ---
EXPERIMENT_NAME="triplet_mtr"
RUN_GROUP_NAME="rq1_lete_multi"
SAFE_TASK_NAME="${TASK_NAME//\//_}"
ACCELERATOR="auto"
PRECISION="32"
STRATEGY="auto"
DEVICES=1
BATCH_SIZE=64
TOKEN_DIM=128
LOGGER="csv"

RUN_OUTPUT_DIR="results/${RUN_GROUP_NAME}/${EXPERIMENT_NAME}/${SAFE_TASK_NAME}/lete_${LETE_VARIANT}_seed${SEED}_epochs${EPOCHS}"
mkdir -p "$RUN_OUTPUT_DIR"

# --- Run ---
echo "🔧 Training LeTE-${LETE_VARIANT} on $TASK_NAME | Seed=$SEED | Epochs=$EPOCHS"
echo "   LeTE Params: p_fourier=$P_FOURIER, fourier_k=$FOURIER_K, spline_hidden_dim=$SPLINE_HIDDEN_DIM"

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
  ++model.input_encoder.lete_p_fourier=${P_FOURIER} \
  ++model.input_encoder.lete_fourier_k=${FOURIER_K} \
  ++model.input_encoder.lete_spline_hidden_dim=${SPLINE_HIDDEN_DIM} \
  ++model.input_encoder.lete_use_layernorm=true \
  hydra.searchpath="[pkg://meds_torch.configs,meds-torch/MIMICIV_INDUCTIVE_EXPERIMENTS/configs/meds-torch-configs]"

exit 0 