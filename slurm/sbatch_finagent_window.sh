#!/usr/bin/env bash
#SBATCH --job-name=finsaber-finagent
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_logs/%x/%A_%a.out
#SBATCH --error=slurm_logs/%x/%A_%a.err

set -euo pipefail

TASK_FILE=${TASK_FILE:?Set TASK_FILE to the generated TSV manifest.}
OUT_ROOT=${OUT_ROOT:-runs/finagent_gpt4o_mini}
MODEL_ID=${MODEL_ID:-gpt-4o-mini}
CONDA_ENV=${CONDA_ENV:-trading}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script is intended to run as a Slurm array job." >&2
  exit 2
fi

mkdir -p "slurm_logs/${SLURM_JOB_NAME:-finsaber-finagent}"

# TASK_FILE has a header. Array index 1 maps to the first data row.
row=$(awk -F'\t' -v idx="$SLURM_ARRAY_TASK_ID" 'NR == idx + 1 {print}' "$TASK_FILE")
if [[ -z "$row" ]]; then
  echo "No manifest row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

IFS=$'\t' read -r setup strat_config_path date_from date_to tickers <<< "$row"

echo "Task ${SLURM_ARRAY_TASK_ID}: setup=${setup} window=${date_from}_${date_to} tickers=${tickers} model=${MODEL_ID}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  echo "conda not found on PATH. Load your cluster's Anaconda/Miniconda module before sbatch." >&2
  exit 2
fi

# Keep secrets outside Git. The runner and FinAgent provider will read these variables.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

python slurm/finagent_one_window.py \
  --setup "$setup" \
  --date-from "$date_from" \
  --date-to "$date_to" \
  --strat-config-path "$strat_config_path" \
  --tickers "$tickers" \
  --model-id "$MODEL_ID" \
  --output-root "$OUT_ROOT"
