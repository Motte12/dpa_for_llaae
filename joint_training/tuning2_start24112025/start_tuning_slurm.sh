#!/bin/bash
#SBATCH --job-name=dpa-tuning
#SBATCH --array=0-863%52   # 864 configs total; run max 52 at a time
#SBATCH --output=/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_tuning2_starting24112025_v3_data/out/slurm-%A_%a.out
#SBATCH --error=/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_tuning2_starting24112025_v3_data/err/slurm-%A_%a.err
#SBATCH --partition=paula
#SBATCH --gpus=a30:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=1-23:00:00

# safer bash settings, but we handle errors explicitly (no `set -e`)
set -u
set -o pipefail

# Directory for per-task status files
STATUS_DIR=/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_tuning2_starting24112025_v3_data/status
mkdir -p "$STATUS_DIR"

echo "Starting task ${SLURM_ARRAY_TASK_ID} at $(date)"

########################
# 1) Hyperparameters
########################

# model
latent_dims=(10 20 50)
encoder=(learnable PCA)
hidden_dim_NNs=(50 100)
num_layers_NNs=(4 6)
noise_dim_dec=(5 10 20)

# latent map
hidden_dim_lm=(20 50)
noise_dim_lm=(20 100)

# training
lambda=(0 0.5 1)

########################
# 2) Build full list of configs
########################

configs=()
for ld in "${latent_dims[@]}"; do
  for enc in "${encoder[@]}"; do
    for hdn in "${hidden_dim_NNs[@]}"; do
      for nln in "${num_layers_NNs[@]}"; do
        for ndd in "${noise_dim_dec[@]}"; do
          for hdl in "${hidden_dim_lm[@]}"; do
            for ndl in "${noise_dim_lm[@]}"; do
              for lamb in "${lambda[@]}"; do
                configs+=("$ld,$enc,$hdn,$nln,$ndd,$hdl,$ndl,$lamb")
              done
            done
          done
        done
      done
    done
  done
done

# Sanity check (optional): ensure SLURM_ARRAY_TASK_ID is in range
num_configs=${#configs[@]}
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= num_configs )); then
  echo "Error: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range (num_configs=${num_configs})" >&2
  exit 1
fi

########################
# 3) Extract config for this task
########################

IFS=',' read -r ld enc hdn nln ndd hdl ndl lambd \
  <<< "${configs[$SLURM_ARRAY_TASK_ID]}"

echo "TASK ${SLURM_ARRAY_TASK_ID} → latent_dim=$ld, encoder=$enc, hidden_dim_NN=$hdn, num_layers_NN=$nln, noise_dim_dec=$ndd, hidden_dim_lm=$hdl, noise_dim_lm=$ndl, lambda=$lambd"

# Per-task status file
TASK_STATUS_FILE="${STATUS_DIR}/task_${SLURM_ARRAY_TASK_ID}.status"
{
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  echo "latent_dim=${ld}"
  echo "encoder=${enc}"
  echo "hidden_dim_NN=${hdn}"
  echo "num_layers_NN=${nln}"
  echo "noise_dim_dec=${ndd}"
  echo "hidden_dim_lm=${hdl}"
  echo "noise_dim_lm=${ndl}"
  echo "lambda=${lambd}"
  echo "start_time=$(date)"
} > "$TASK_STATUS_FILE"

########################
# 4) Run training and capture exit code
########################

~/.conda/envs/dpa/bin/python ../train_joint_dpa_automated.py \
    --settings_file dpa_tune2_settings.json \
    --epochs 100 \
    --encoder "$enc" \
    --in_dim 648 \
    --latent_dim "$ld" \
    --num_layer "$nln" \
    --hidden_dim "$hdn" \
    --noise_dim_dec "$ndd" \
    --noise_dim_lm "$ndl" \
    --num_layer_lm 2 \
    --hidden_dim_lm "$hdl" \
    --lam "$lambd"

exit_code=$?

if [ "$exit_code" -eq 0 ]; then
  echo "status=SUCCESS" >> "$TASK_STATUS_FILE"
  echo "end_time=$(date)" >> "$TASK_STATUS_FILE"
  echo "Task ${SLURM_ARRAY_TASK_ID} completed successfully."
else
  echo "status=FAIL" >> "$TASK_STATUS_FILE"
  echo "end_time=$(date)" >> "$TASK_STATUS_FILE"
  echo "exit_code=${exit_code}" >> "$TASK_STATUS_FILE"
  echo "Task ${SLURM_ARRAY_TASK_ID} FAILED with exit code ${exit_code}" >&2
fi

exit "$exit_code"
