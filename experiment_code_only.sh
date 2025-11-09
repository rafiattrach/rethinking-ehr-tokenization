#!/bin/bash

# === Code-Only Ablation Batch Launcher ===
# Runs all 4 tasks with code-only triplet variant × 10 seeds (0-9)
# Tests the impact of removing both temporal and value information from triplet representations
# Simple logging with timing and system details

LOG_FILE="code_only_ablation_run_log.txt"
touch "$LOG_FILE"

TASKS=(
  "mortality/in_hospital/first_24h"
  "mortality/in_icu/first_24h"
  "mortality/post_hospital_discharge/1y"
  "readmission/30d"
)

VARIANT="code_only"
EPOCHS=10
SEEDS=(0 1 2 3 4 5 6 7 8 9)  # 10 seeds as requested

# --- System Info ===
BATCH_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
HOSTNAME=$(hostname)
PYTHON_VERSION=$(python --version 2>&1)

echo "=== Starting Code-Only Ablation Experiments - $BATCH_START_TIME ===" | tee -a "$LOG_FILE"
echo "Host: $HOSTNAME | Python: $PYTHON_VERSION" | tee -a "$LOG_FILE"
echo "Total runs: $((${#TASKS[@]} * ${#SEEDS[@]}))" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

RUN_COUNT=0
TOTAL_RUNS=$((${#TASKS[@]} * ${#SEEDS[@]}))

for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN_COUNT=$((RUN_COUNT + 1))
    RUN_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    RUN_START_TIMESTAMP=$(date +%s)
    
    echo "[$RUN_COUNT/$TOTAL_RUNS] Launching: Task='${TASK}' Variant='${VARIANT}' Seed=${SEED} - Started: $RUN_START_TIME"
    
    # --- Activate environment ---
    source .rqvenv/bin/activate

    # --- Set config ---
    EXPERIMENT_NAME="triplet_mtr" # This uses the code_only encoder via a branch
    RUN_GROUP_NAME="code_only_ablation_multi"
    SAFE_TASK_NAME="${TASK//\//_}"
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
    meds-torch-train \
      experiment=${EXPERIMENT_NAME} \
      paths.data_dir=triplet_tensors \
      paths.meds_cohort_dir=MEDS_cohort \
      paths.output_dir=${RUN_OUTPUT_DIR} \
      data.task_name=${TASK} \
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

    EXIT_CODE=$?
    
    RUN_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    RUN_DURATION=$(($(date +%s) - RUN_START_TIMESTAMP))
    RUN_DURATION_FORMATTED=$(printf "%02d:%02d:%02d" $((RUN_DURATION/3600)) $((RUN_DURATION%3600/60)) $((RUN_DURATION%60)))
    
    # Log the result
    if [[ $EXIT_CODE -eq 0 ]]; then
      echo "✅ SUCCESS: $TASK, $VARIANT, seed=$SEED | Duration: $RUN_DURATION_FORMATTED | Started: $RUN_START_TIME | Ended: $RUN_END_TIME | Python: $PYTHON_VERSION | Host: $HOSTNAME" | tee -a "$LOG_FILE"
    else
      echo "❌ FAILED: $TASK, $VARIANT, seed=$SEED | Duration: $RUN_DURATION_FORMATTED | Started: $RUN_START_TIME | Ended: $RUN_END_TIME | Python: $PYTHON_VERSION | Host: $HOSTNAME" | tee -a "$LOG_FILE"
    fi
    echo ""
  done
done

echo "=== Completed all Code-Only Ablation Experiments - $(date) ===" | tee -a "$LOG_FILE" 