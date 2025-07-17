#!/bin/bash

# Script to train the autocorrect model using parallel processing
# Optimized for a 48-core machine with 128GB RAM

# Default paths - update these as needed
DATA_PATH="./data/optimized_samples.pt"
WORD2VEC_PATH="./data/GoogleNews-vectors-negative300.bin"
MODEL_PATH="./models/char_autocorrect_parallel.pt"

# Training parameters
BATCH_SIZE=2048  # Larger batch size for 128GB RAM
NUM_WORKERS=40   # Use 40 cores, leaving 8 for system processes
EPOCHS=5

# Create models directory if it doesn't exist
mkdir -p ./models

# Print system information
echo "=== System Information ==="
echo "CPU cores: $(nproc)"
free -h | grep "Mem:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Run the training script
echo "=== Starting Parallel Training ==="
python run_3_parallel_processing.py \
  --train \
  --data "$DATA_PATH" \
  --word2vec "$WORD2VEC_PATH" \
  --model "$MODEL_PATH" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --epochs "$EPOCHS"

echo "=== Training Complete ==="
echo "Model saved to $MODEL_PATH"
