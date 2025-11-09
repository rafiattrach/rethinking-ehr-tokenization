#!/bin/bash

# === RQ3 Flexible TextCode Experiments for SageMaker ===
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_rq3_flexible_textcode_experiments.sh "experiment_name"
# Example: CUDA_VISIBLE_DEVICES=0 bash run_rq3_flexible_textcode_experiments.sh "enhanced_mapping_tinybert_frozen"

# --- Environment Setup ---
export CODECARBON_DISABLE=1
export CODECARBON_OFFLINE=1
export CODECARBON_LOG_LEVEL=ERROR
export TOKENIZERS_PARALLELISM=false


# --- Parse Arguments ---
if [ $# -ne 1 ]; then
    echo "Usage: CUDA_VISIBLE_DEVICES=0 bash run_rq3_flexible_textcode_experiments.sh \"experiment_name\""
    echo ""
    echo "Available experiments:"
    echo "  baseline_original_mapping_tinybert_trainable"
    echo "  enhanced_mapping_tinybert_trainable"
    echo "  original_mapping_tinybert_frozen"
    echo "  enhanced_mapping_tinybert_frozen"
    echo "  enhanced_mapping_bigbert_frozen"
    echo "  enhanced_mapping_qwen3_frozen"
    echo "  original_mapping_tinybert_frozen_code_only"
    echo "  original_mapping_tinybert_trainable_code_only"
    exit 1
fi

EXPERIMENT_NAME="$1"

# --- Validate Experiment Name ---
case $EXPERIMENT_NAME in
    "baseline_original_mapping_tinybert_trainable"|"enhanced_mapping_tinybert_trainable"|"original_mapping_tinybert_frozen"|"enhanced_mapping_tinybert_frozen"|"enhanced_mapping_bigbert_frozen"|"enhanced_mapping_qwen3_frozen"|"original_mapping_tinybert_frozen_code_only"|"original_mapping_tinybert_trainable_code_only")
        echo "✅ Valid experiment: $EXPERIMENT_NAME"
        ;;
    *)
        echo "❌ Invalid experiment: $EXPERIMENT_NAME"
        echo "Available experiments:"
        echo "  baseline_original_mapping_tinybert_trainable"
        echo "  enhanced_mapping_tinybert_trainable"
        echo "  original_mapping_tinybert_frozen"
        echo "  enhanced_mapping_tinybert_frozen"
        echo "  enhanced_mapping_bigbert_frozen"
        echo "  enhanced_mapping_qwen3_frozen"
        echo "  original_mapping_tinybert_frozen_code_only"
        echo "  original_mapping_tinybert_trainable_code_only"
        exit 1
        ;;
esac

# --- Set Experiment Parameters ---
case $EXPERIMENT_NAME in
    "baseline_original_mapping_tinybert_trainable")
        MODEL="nlpie/tiny-clinicalbert"
        MAPPING="triplet_tensors/metadata/codes.parquet"
        FROZEN="false"
        USE_TIME="true"
        USE_VALUE="true"
        DESCRIPTION="Baseline: Original mapping + TinyBERT trainable"
        ;;
    "enhanced_mapping_tinybert_trainable")
        MODEL="nlpie/tiny-clinicalbert"
        MAPPING="meds-torch/mapping/meds_triplet_descriptions.csv"
        FROZEN="false"
        USE_TIME="true"
        USE_VALUE="true"
        DESCRIPTION="Enhanced: Enhanced mapping + TinyBERT trainable"
        ;;
    "original_mapping_tinybert_frozen")
        MODEL="nlpie/tiny-clinicalbert"
        MAPPING="triplet_tensors/metadata/codes.parquet"
        FROZEN="true"
        USE_TIME="true"
        USE_VALUE="true"
        DESCRIPTION="Original: Original mapping + TinyBERT frozen"
        ;;
    "enhanced_mapping_tinybert_frozen")
        MODEL="nlpie/tiny-clinicalbert"
        MAPPING="meds-torch/mapping/meds_triplet_descriptions.csv"
        FROZEN="true"
        USE_TIME="true"
        USE_VALUE="true"
        DESCRIPTION="Enhanced: Enhanced mapping + TinyBERT frozen"
        ;;
    "enhanced_mapping_bigbert_frozen")
        MODEL="thomas-sounack/BioClinical-ModernBERT-large"
        MAPPING="meds-torch/mapping/meds_triplet_descriptions.csv"
        FROZEN="true"
        USE_TIME="true"
        USE_VALUE="true"
        DESCRIPTION="BigBERT: Enhanced mapping + BigBERT frozen"
        ;;
    "enhanced_mapping_qwen3_frozen")
        MODEL="Qwen/Qwen3-Embedding-0.6B"
        MAPPING="meds-torch/mapping/meds_triplet_descriptions.csv"
        FROZEN="true"
        USE_TIME="true"
        USE_VALUE="true"
        DESCRIPTION="Qwen3: Enhanced mapping + Qwen3 frozen"
        ;;
    "original_mapping_tinybert_frozen_code_only")
        MODEL="nlpie/tiny-clinicalbert"
        MAPPING="triplet_tensors/metadata/codes.parquet"
        FROZEN="true"
        USE_TIME="false"
        USE_VALUE="false"
        DESCRIPTION="Code-only: Original mapping + TinyBERT frozen (no time/value)"
        ;;
    "original_mapping_tinybert_trainable_code_only")
        MODEL="nlpie/tiny-clinicalbert"
        MAPPING="triplet_tensors/metadata/codes.parquet"
        FROZEN="false"
        USE_TIME="false"
        USE_VALUE="false"
        DESCRIPTION="Code-only: Original mapping + TinyBERT trainable (no time/value)"
        ;;
esac

# --- Check and Switch to Correct Branch ---
# This check is disabled for the public repository to ensure portability.
# The code is self-contained in the main branch.

# --- Activate Environment ---
if [ -d ".rqvenv" ]; then
    source .rqvenv/bin/activate
    echo "✅ Activated .rqvenv environment"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Activated .venv environment"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Activated venv environment"
else
    echo "❌ No virtual environment found (.rqvenv, .venv, or venv)"
    exit 1
fi

# --- Set Configuration ---
EXPERIMENT_CONFIG="rq3_flexible_textcode_mtr"
RUN_GROUP_NAME="rq3_flexible_textcode_multi"
ACCELERATOR="auto"
PRECISION="32"
STRATEGY="auto"
DEVICES=1
BATCH_SIZE=64
TOKEN_DIM=128
LOGGER="many_loggers"

# --- Setup Logging ---
LOG_FILE="rq3_${EXPERIMENT_NAME}_progress.txt"
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
GPU_ID=${CUDA_VISIBLE_DEVICES:-"unknown"}

# Create log file and write header
echo "=== RQ3 Experiment Progress Log ===" > "$LOG_FILE"
echo "Experiment: $EXPERIMENT_NAME" >> "$LOG_FILE"
echo "Started: $START_TIME" >> "$LOG_FILE"
echo "GPU: $GPU_ID" >> "$LOG_FILE"
echo "Branch: $CURRENT_BRANCH" >> "$LOG_FILE"
echo "Total Runs: 40 (4 tasks × 10 seeds)" >> "$LOG_FILE"
echo "Model: $MODEL" >> "$LOG_FILE"
echo "Mapping: $MAPPING" >> "$LOG_FILE"
echo "Frozen: $FROZEN" >> "$LOG_FILE"
echo "==================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# --- Tasks and Seeds ---
TASKS=("mortality/in_hospital/first_24h" "mortality/in_icu/first_24h" "mortality/post_hospital_discharge/1y" "readmission/30d")
SEEDS=(0 1 2 3 4 5 6 7 8 9)
EPOCHS=10

# --- Initialize Counter ---
RUN_COUNTER=0
TOTAL_RUNS=$((${#TASKS[@]} * ${#SEEDS[@]}))
echo "📊 Total runs for this experiment: $TOTAL_RUNS"

# --- Run All Tasks and Seeds ---
for TASK in "${TASKS[@]}"; do
    SAFE_TASK_NAME="${TASK//\//_}"
    
    for SEED in "${SEEDS[@]}"; do
        RUN_COUNTER=$((RUN_COUNTER + 1))
        RUN_OUTPUT_DIR="results/${RUN_GROUP_NAME}/triplet_mtr_${EXPERIMENT_NAME}/${SAFE_TASK_NAME}/seed${SEED}_epochs${EPOCHS}"
        mkdir -p "$RUN_OUTPUT_DIR"
        
        # Create descriptive WandB run name
        WANDB_RUN_NAME="rq3_${EXPERIMENT_NAME}_${SAFE_TASK_NAME}_seed${SEED}"
        
        echo ""
        echo "🚀 ========================================"
        echo "🚀 Starting: $DESCRIPTION"
        echo "📁 Task: $TASK"
        echo "🎲 Seed: $SEED"
        echo "📊 Epochs: $EPOCHS"
        echo "💾 Output: $RUN_OUTPUT_DIR"
        echo "🔧 Model: $MODEL"
        echo "🗺️  Mapping: $MAPPING"
        echo "❄️  Frozen: $FROZEN"
        PERCENTAGE=$((RUN_COUNTER * 100 / TOTAL_RUNS))
        echo "📈 Progress: $RUN_COUNTER/$TOTAL_RUNS ($PERCENTAGE%)"
        echo "🚀 ========================================"
        echo ""
        
        # Record start time for this run
        RUN_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        
        meds-torch-train \
            experiment=${EXPERIMENT_CONFIG} \
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
            ++model.input_encoder.code_embedder=${MODEL} \
            ++model.input_encoder.code_tokenizer=${MODEL} \
            ++model.input_encoder.code_metadata_fp=${MAPPING} \
            ++model.input_encoder.freeze_model=${FROZEN} \
            ++model.input_encoder.use_time=${USE_TIME} \
            ++model.input_encoder.use_value=${USE_VALUE} \
            ++logger.wandb.group="rq3_${EXPERIMENT_NAME}" \
            ++logger.wandb.name="${WANDB_RUN_NAME}" \
            ++logger.wandb.project="rq3_${EXPERIMENT_NAME}" \
            hydra.searchpath="[pkg://meds_torch.configs,meds-torch/MIMICIV_INDUCTIVE_EXPERIMENTS/configs/meds-torch-configs]"
        
        # Record end time and calculate duration
        RUN_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        RUN_DURATION=$(($(date +%s) - $(date -d "$RUN_START_TIME" +%s)))
        
        # Check if training completed successfully
        if [ $? -eq 0 ]; then
            STATUS="✅ SUCCESS"
            echo "✅ ========================================"
            echo "✅ SUCCESS: $DESCRIPTION | $TASK | Seed $SEED"
            echo "✅ Duration: ${RUN_DURATION}s"
            echo "✅ Completed: $RUN_END_TIME"
            echo "✅ Progress: $RUN_COUNTER/$TOTAL_RUNS"
            echo "✅ ========================================"
        else
            STATUS="❌ FAILED"
            echo "❌ ========================================"
            echo "❌ FAILED: $DESCRIPTION | $TASK | Seed $SEED"
            echo "❌ Duration: ${RUN_DURATION}s"
            echo "❌ Failed: $RUN_END_TIME"
            echo "❌ Progress: $RUN_COUNTER/$TOTAL_RUNS"
            echo "❌ ========================================"
            echo "🚨 Continuing with next experiment..."
        fi
        
        # Log one comprehensive line to progress file with emoji
        echo "[$RUN_END_TIME] $STATUS | Run:$RUN_COUNTER/$TOTAL_RUNS | GPU:$GPU_ID | Branch:$CURRENT_BRANCH | Exp:$EXPERIMENT_NAME | Task:$TASK | Seed:$SEED | Duration:${RUN_DURATION}s | Model:$MODEL | Frozen:$FROZEN | Output:$RUN_OUTPUT_DIR" >> "$LOG_FILE"
        
        echo "---"
    done
done

echo ""
echo "🎉 All experiments completed for: $DESCRIPTION"
echo "📁 Results saved in: results/${RUN_GROUP_NAME}/triplet_mtr_${EXPERIMENT_NAME}/"
echo "📊 Progress log: $LOG_FILE"

# Log final summary
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_DURATION=$(($(date +%s) - $(date -d "$START_TIME" +%s)))
echo "[$END_TIME] 🎉 COMPLETED | GPU:$GPU_ID | Branch:$CURRENT_BRANCH | Exp:$EXPERIMENT_NAME | TotalDuration:${TOTAL_DURATION}s | AllTasks:${#TASKS[@]} | AllSeeds:${#SEEDS[@]} | TotalRuns:40" >> "$LOG_FILE" 