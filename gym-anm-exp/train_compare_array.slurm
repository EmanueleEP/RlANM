#!/bin/bash
#SBATCH --job-name=anm-seeds
#SBATCH --array=0-14%2
#SBATCH --output=slurm_logs/compare_%A_%a.out
#SBATCH --error=slurm_logs/compare_%A_%a.err
#SBATCH --partition=allgroups
#SBATCH --time=35:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

SEEDS=(1 12 123 1234 12345)
TASK_ID="${SLURM_ARRAY_TASK_ID}"
SEED_INDEX=$((TASK_ID % ${#SEEDS[@]}))
SEED="${SEEDS[$SEED_INDEX]}"

METHODS=(original lagrangian lagrangian_pid)
METHOD="${METHODS[$((TASK_ID / ${#SEEDS[@]}))]}"

PROJECT_DIR="/home/paseemanue/Desktop/rl_tesi/RlANM/gym-anm-exp"
VENV_ACTIVATE="/home/paseemanue/Desktop/rl_tesi/project/bin/activate"
EXPERIMENT_ID="${EXPERIMENT_ID:-experiment_01}"
TRAIN_STEPS="${TRAIN_STEPS:-3000000}"
EVAL_FREQ="${EVAL_FREQ:-200000}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
# PROFILE_FILE=none trains on the built-in ANM6Easy profiles (as in the paper).
# Kept separate: the "}" of {split} would end a ${PROFILE_FILE:-...} expansion early.
DEFAULT_PROFILE_FILE="${PROJECT_DIR}/../../Datasets/processed/profiles_{split}.npz"
PROFILE_FILE="${PROFILE_FILE:-${DEFAULT_PROFILE_FILE}}"
PROFILE_ARGS=()
if [[ "${PROFILE_FILE}" != "none" ]]; then
    PROFILE_ARGS=(--profile-file "${PROFILE_FILE}")
fi
RUN_DIR="${PROJECT_DIR}/results/${EXPERIMENT_ID}/${METHOD}/seed_${SEED}"

source "${VENV_ACTIVATE}"
cd "${PROJECT_DIR}"

# Avoid CPU oversubscription inside each independent training process.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun python -m rl_agents.train PPO \
    --seed "${SEED}" \
    --run-dir "${RUN_DIR}" \
    --steps "${TRAIN_STEPS}" \
    --eval-freq "${EVAL_FREQ}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --step-log-every 5000 \
    --method "${METHOD}" \
    "${PROFILE_ARGS[@]}"
