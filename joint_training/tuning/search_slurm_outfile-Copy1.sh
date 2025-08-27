#!/bin/bash

#LOG_DIR="/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_tuning_output"
#if ! cd "$LOG_DIR"; then
#  echo "ERROR: Could not cd into $LOG_DIR" >&2
#  exit 1#
#fi

#for f in slurm-*.out; do
#  if grep -q "latent_dim=10" "$f" \
#     && grep -q "encoder=PCA" "$f" \
#     && grep -q "lambda=0.5" "$f"; then
#    echo $f
#  fi
#done

#echo "done"


# ------------------------------------------------------------------
# Usage: $0 LOG_DIR LATENT_DIM ENCODER LAMBDA \
#          HIDDEN_DIM_NNs NUM_LAYERS_NNs NOISE_DIM_DEC \
#          HIDDEN_DIM_LM NOISE_DIM_LM
#
# Example:
#   bash $0 /path/to/logs 10 PCA 0.5 \
#            100 6 10 20 20
# ------------------------------------------------------------------

if [ "$#" -ne 9 ]; then
  echo "ERROR: Wrong number of arguments." >&2
  echo "Usage: $0 LOG_DIR LATENT_DIM ENCODER LAMBDA \\" >&2
  echo "           HIDDEN_DIM_NNs NUM_LAYERS_NNs NOISE_DIM_DEC \\" >&2
  echo "           HIDDEN_DIM_LM NOISE_DIM_LM" >&2
  exit 1
fi

LOG_DIR="$1"
LATENT_DIM="$2"
ENCODER="$3"
LAMBDA="$4"
HIDDEN_DIM_NNs="$5"
NUM_LAYERS_NNs="$6"
NOISE_DIM_DEC="$7"
HIDDEN_DIM_LM="$8"
NOISE_DIM_LM="$9"

# --- Echo all incoming arguments ---
echo "LOG_DIR         = $LOG_DIR"
echo "LATENT_DIM      = $LATENT_DIM"
echo "ENCODER         = $ENCODER"
echo "LAMBDA          = $LAMBDA"
echo "------------------------------------"



# 1) cd into the logs directory
if ! cd "$LOG_DIR"; then
  echo "ERROR: Could not cd into $LOG_DIR" >&2
  exit 1
fi

# 2) Loop over slurm-*.out and grep using our variables
for f in slurm-*.out; do
  if grep -q "latent_dim=10"      "$f" && \
     grep -q "encoder=PCA"           "$f" && \
     grep -q "lambda=0.5"             "$f"; then #&& \
     #grep -q "hidden_dim_NNs=${HIDDEN_DIM_NNs}" "$f" && \
     #grep -q "num_layers_NNs=${NUM_LAYERS_NNs}" "$f" && \
     #grep -q "noise_dim_dec=${NOISE_DIM_DEC}"   "$f" && \
     #grep -q "hidden_dim_lm=${HIDDEN_DIM_LM}"   "$f" && \
     #grep -q "noise_dim_lm=${NOISE_DIM_LM}"     "$f"; then
    echo "$f"
  fi
done

echo "done"
