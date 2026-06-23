#!/usr/bin/env bash
# Generate a one-window-per-job manifest and submit it as a Slurm array.

set -euo pipefail

SUITE=${SUITE:-paper_rest}
OUT_ROOT=${OUT_ROOT:-runs/finagent_gpt4o_mini}
MODEL_ID=${MODEL_ID:-gpt-4o-mini}
CONDA_ENV=${CONDA_ENV:-trading}
MAX_PARALLEL=${MAX_PARALLEL:-24}
MANIFEST=${MANIFEST:-${OUT_ROOT}/manifests/${SUITE}.tsv}

mkdir -p "${OUT_ROOT}/manifests" slurm_logs

python slurm/make_finagent_window_manifest.py --suite "$SUITE" --output "$MANIFEST"

num_tasks=$(( $(wc -l < "$MANIFEST") - 1 ))
if [[ "$num_tasks" -le 0 ]]; then
  echo "No tasks generated in ${MANIFEST}" >&2
  exit 2
fi

echo "Submitting ${num_tasks} FinAgent tasks from ${MANIFEST}"
echo "Suite=${SUITE} Model=${MODEL_ID} Output=${OUT_ROOT} MaxParallel=${MAX_PARALLEL} CondaEnv=${CONDA_ENV}"

sbatch \
  --array="1-${num_tasks}%${MAX_PARALLEL}" \
  --export=ALL,TASK_FILE="$MANIFEST",OUT_ROOT="$OUT_ROOT",MODEL_ID="$MODEL_ID",CONDA_ENV="$CONDA_ENV" \
  slurm/sbatch_finagent_window.sh
