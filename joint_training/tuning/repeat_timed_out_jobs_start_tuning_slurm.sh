#!/bin/bash
#SBATCH --job-name=missed-jobs-dpa-tuning
#SBATCH --array=0-863%24
#SBATCH --output=/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_tuning_output_missed_jobs/slurm-%A_%a.out
#SBATCH --partition=paula
#SBATCH --gpus=a30:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00

# ------------------------------------------------------------------
# This script runs only those array‐tasks whose IDs are listed in timeout_jobs.txt
# Usage (inside the submission directory):
#   sbatch this_script.sh
# Make sure timeout_jobs.txt lives somewhere accessible (update ALLOWED_FILE)
# ------------------------------------------------------------------

# 1) Define hyperparameter arrays
latent_dims=(10 20 50)
encoder=(learnable PCA)
hidden_dim_NNs=(50 100)
num_layers_NNs=(4 6)
noise_dim_dec=(5 10 20)
hidden_dim_lm=(20 50)
noise_dim_lm=(20 100)
lambda=(0 0.5 1)

# 2) Build the full list of comma‐separated configs
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

# 3) Unpack the parameters for this array‐task
IFS=',' read -r ld enc hdn nln ndd hdl ndl lambd \
  <<< "${configs[$SLURM_ARRAY_TASK_ID]}"

# 4) Debug echo
echo "TASK ${SLURM_ARRAY_TASK_ID} → latent_dim=$ld, encoder=$enc, hidden_dim_NN=$hdn, num_layers_NN=$nln, noise_dim_dec=$ndd, hidden_dim_lm=$hdl, noise_dim_lm=$ndl, lambda=$lambd"

# 5) Load allowed indices from file (stripping everything before '_')
ALLOWED_FILE="timeout_jobs.txt"   # <-- update this path
declare -A allowed
while read -r jobid; do
  idx="${jobid#*_}"       # remove up through the underscore
  allowed["$idx"]=1
done < "$ALLOWED_FILE"

# 6) If this task’s index isn’t in the allowed list, skip it
if [[ -z "${allowed[$SLURM_ARRAY_TASK_ID]}" ]]; then
  echo "Skipping SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} (not in allowed list)"
  exit 0
fi

# 6) Invoke the training script with the selected hyperparameters
~/.conda/envs/dpa/bin/python ../train_joint_dpa_automated.py \
    --settings_file repeat_timed_out_jobs_dpa_tune_settings.json \
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

echo "done"
