#!/bin/bash
# Example SLURM job for Hygon DCU/ROCm clusters.
# Adjust partition/account/module paths for your environment.

#SBATCH -J patent_parse_dcu
#SBATCH -p dcu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:4
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

mkdir -p logs

# ---------- Environment ----------
# Example module commands (replace with your cluster's module names)
# module purge
# module load rocm/6.0
# module load python/3.11

# Activate your runtime env
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate mineru

# Local cache/scratch to reduce shared filesystem IO pressure
export JOB_SCRATCH="${SLURM_TMPDIR:-/tmp}/${USER}/patent_parse_${SLURM_JOB_ID}"
mkdir -p "${JOB_SCRATCH}"

export TMPDIR="${JOB_SCRATCH}/tmp"
export TRANSFORMERS_CACHE="${JOB_SCRATCH}/hf_cache"
export HF_HOME="${JOB_SCRATCH}/hf_home"
export MIOPEN_USER_DB_PATH="${JOB_SCRATCH}/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="${JOB_SCRATCH}/miopen_cache"
mkdir -p "${TMPDIR}" "${TRANSFORMERS_CACHE}" "${HF_HOME}" "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}"

# Explicit visible devices for this job (one worker per card recommended)
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HSA_VISIBLE_DEVICES=0,1,2,3

# ---------- Inputs ----------
INPUT_ROOT="/path/to/input_root"
OUTPUT_ROOT="/path/to/output_root_${SLURM_JOB_ID}"
WIPO_META="/path/to/wipo_metadata_cache.json"
BIBLIO_META="/path/to/biblio_metadata_cache.json"

# ---------- Run ----------
srun python parse.py \
  -i "${INPUT_ROOT}" \
  -o "${OUTPUT_ROOT}" \
  -g 0,1,2,3 \
  -w 4 \
  -m auto \
  --wipo-metadata "${WIPO_META}" \
  --biblio-metadata "${BIBLIO_META}" \
  --parser-version "dcu-prod-v1"

# Optional: sync OUTPUT_ROOT to long-term storage here.
# rsync -a "${OUTPUT_ROOT}/" /path/to/final_storage/

