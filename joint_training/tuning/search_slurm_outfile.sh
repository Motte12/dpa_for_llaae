#!/bin/bash

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
echo "HIDDEN_DIM_NNs  = $HIDDEN_DIM_NNs"
echo "NUM_LAYERS_NNs  = $NUM_LAYERS_NNS"
echo "NOISE_DIM_DEC   = $NOISE_DIM_DEC"
echo "HIDDEN_DIM_LM   = $HIDDEN_DIM_LM"
echo "NOISE_DIM_LM    = $NOISE_DIM_LM"
echo "------------------------------------"

LOG_DIR="/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_tuning_output"
if ! cd "$LOG_DIR"; then
  echo "ERROR: Could not cd into $LOG_DIR" >&2
  exit 1#
fi

for f in slurm-*.out; do
  if grep -q "latent_dim=${LATENT_DIM}" "$f" \
     && grep -q "encoder=${ENCODER}" "$f" \
     && grep -q "lambda=${LAMBDA}" "$f" \
     && grep -q "hidden_dim_NN=${HIDDEN_DIM_NNs}" "$f" \
     && grep -q "num_layers_NN=${NUM_LAYERS_NNs}" "$f" \
     && grep -q "noise_dim_dec=${NOISE_DIM_DEC}"   "$f" \
     && grep -q "hidden_dim_lm=${HIDDEN_DIM_LM}"   "$f" \
     && grep -q "noise_dim_lm=${NOISE_DIM_LM}"     "$f"; then
    echo $f
  fi
done

