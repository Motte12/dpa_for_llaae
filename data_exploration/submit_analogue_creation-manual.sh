#!/bin/bash
# Optional: print job info
echo "Starting job on $(hostname) at $(date)"
echo "Running from: $(pwd)"


# Run the script 
~/.conda/envs/dpa/bin/python create_analogues_parallel_v4_data.py

echo "Job finished at $(date)"
