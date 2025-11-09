#!/bin/bash
# Runs the BASELINE (CVE) model for multiple seeds (replication params)

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Activate Environment ---
VENV_PATH=".rqvenv/bin/activate" # Assumes script runs from project root
if [ -f "$VENV_PATH" ]; then
    echo "Activating venv environment: $VENV_PATH"
    source "$VENV_PATH"
else
    echo "ERROR: Virtual environment activation script not found at $VENV_PATH"
    exit 1
fi
if ! command -v meds-torch-train &> /dev/null; then
    echo "ERROR: 'meds-torch-train' command not found after venv activation."
    exit 1
fi
echo "Environment activated successfully."

# --- Configuration ---
TASK_NAME="mortality/in_hospital/first_24h" # Task to train on
EXPERIMENT_NAME="triplet_mtr"                # Baseline experiment config
RUN_GROUP_NAME="rq1_baseline_multi"          # <--- Name for this group of runs
ACCELERATOR="auto"                         # 'auto', 'mps', 'gpu', 'cpu'.
PRECISION="32"                             # Use '16-mixed' on capable GPUs if desired
STRATEGY="auto"
DEVICES=1
MAX_EPOCHS=10                              # Replication epochs
BATCH_SIZE=64                              # Replication batch size
TOKEN_DIM=128                              # Replication token dimension
# NUM_WORKERS handled conditionally
LOGGER="csv"                               # Logger type
NUM_SEEDS=5                                # <<< Number of seeds to run

# --- Determine Project Root Dynamically ---
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- Define Paths ---
REPO_DIR="${PROJECT_ROOT}/meds-torch"
MEDS_DIR="${PROJECT_ROOT}/MEDS_cohort"
TENSOR_DIR="${PROJECT_ROOT}/triplet_tensors"
RESULTS_BASE_DIR="${PROJECT_ROOT}/results"
CONFIG_DIR="${REPO_DIR}/MIMICIV_INDUCTIVE_EXPERIMENTS/configs"

# --- Determine num_workers ---
EFFECTIVE_ACCELERATOR=$ACCELERATOR
# (Same auto-detection logic as before)
if [[ "$ACCELERATOR" == "auto" ]]; then
    if command -v system_profiler &> /dev/null && system_profiler SPDisplaysDataType | grep -q "Metal Family: Supported"; then EFFECTIVE_ACCELERATOR="mps"; fi
    if command -v nvidia-smi &> /dev/null; then EFFECTIVE_ACCELERATOR="gpu"; fi
    if [[ "$EFFECTIVE_ACCELERATOR" == "auto" ]]; then EFFECTIVE_ACCELERATOR="cpu"; fi
fi
if [[ "$EFFECTIVE_ACCELERATOR" == "mps" ]]; then NUM_WORKERS=0; else NUM_WORKERS=6; fi
echo "Using Effective Accelerator: ${EFFECTIVE_ACCELERATOR} with Num Workers: ${NUM_WORKERS}"

# --- Base Output Directory for this Group ---
SAFE_TASK_NAME="${TASK_NAME//\//_}"
GROUP_OUTPUT_DIR="${RESULTS_BASE_DIR}/${RUN_GROUP_NAME}/${EXPERIMENT_NAME}/${SAFE_TASK_NAME}"
echo "Base output directory for this run group: ${GROUP_OUTPUT_DIR}"
mkdir -p "$GROUP_OUTPUT_DIR"


# --- Loop Over Seeds ---
echo "Starting $NUM_SEEDS runs for Baseline..."
for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
    echo "================================="
    echo "Running Seed: $SEED"
    echo "================================="

    # Construct Timestamped Output Path for this specific seed
    TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')_seed${SEED}
    RUN_OUTPUT_DIR="${GROUP_OUTPUT_DIR}/${TIMESTAMP}"
    TRAIN_OUTPUT_DIR="${RUN_OUTPUT_DIR}"
    TASK_LABEL_DIR="${MEDS_DIR}/tasks"

    # Create output directory for this seed
    mkdir -p "$TRAIN_OUTPUT_DIR"
    echo "Output directory for this seed: $TRAIN_OUTPUT_DIR"

    # --- Save Run Configuration Summary ---
    CONFIG_SUMMARY_FILE="${TRAIN_OUTPUT_DIR}/run_config_summary.txt"
    echo "Saving run configuration to: ${CONFIG_SUMMARY_FILE}"
    {
        echo "Run Group Name: ${RUN_GROUP_NAME}"
        echo "Timestamp: ${TIMESTAMP}"
        echo "Task Name: ${TASK_NAME}"
        echo "Experiment Name: ${EXPERIMENT_NAME}"
        echo "Seed: ${SEED}" # <<< Record the specific seed
        echo "--- Trainer ---"
        echo "Accelerator: ${ACCELERATOR} (Effective: ${EFFECTIVE_ACCELERATOR})"
        echo "Devices: ${DEVICES}"
        echo "Precision: ${PRECISION}"
        echo "Strategy: ${STRATEGY}"
        echo "Max Epochs: ${MAX_EPOCHS}"
        echo "--- Data ---"
        echo "Batch Size: ${BATCH_SIZE}"
        echo "Num Workers: ${NUM_WORKERS}"
        echo "--- Model ---"
        echo "Token Dim: ${TOKEN_DIM}"
        echo "Input Encoder Time: CVE (Baseline)" # Note baseline
        echo "--- Paths ---"
        echo "Output Dir: ${TRAIN_OUTPUT_DIR}"
    } > "$CONFIG_SUMMARY_FILE"

    # --- Construct and Execute the training command ---
    echo "Executing command for Seed $SEED:"
    set -x
    meds-torch-train \
        "experiment=${EXPERIMENT_NAME}" \
        "paths.data_dir=${TENSOR_DIR}" \
        "paths.meds_cohort_dir=${MEDS_DIR}" \
        "paths.output_dir=${TRAIN_OUTPUT_DIR}" \
        "data.task_name=${TASK_NAME}" \
        "data.task_root_dir=${TASK_LABEL_DIR}" \
        "trainer.accelerator=${ACCELERATOR}" \
        "trainer.devices=${DEVICES}" \
        "trainer.precision=${PRECISION}" \
        "trainer.strategy=${STRATEGY}" \
        "logger=${LOGGER}" \
        "seed=${SEED}" \
        "++model.token_dim=${TOKEN_DIM}" \
        "++data.dataloader.batch_size=${BATCH_SIZE}" \
        "++trainer.max_epochs=${MAX_EPOCHS}" \
        "++data.dataloader.num_workers=${NUM_WORKERS}" \
        "hydra.searchpath=[pkg://meds_torch.configs,${CONFIG_DIR}/meds-torch-configs]"
    set +x

    # Check the exit status
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: Run for Seed $SEED failed with exit code $EXIT_CODE!"
        echo "Run configuration saved to: ${CONFIG_SUMMARY_FILE}"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # Decide whether to stop or continue; continue is useful for getting partial results
        # exit 1
    else
        echo "--- Seed $SEED completed successfully. ---"
    fi

done # End of seed loop

echo "---------------------------------"
echo "All $NUM_SEEDS baseline seeds completed."
echo "Results stored under: $GROUP_OUTPUT_DIR"
echo "---------------------------------"

exit 0