#!/usr/bin/env bash
# ==============================================================================
# Wan 2.2 Motion LoRA Training Script (10 Clips - Anal Category)
# ==============================================================================

set -e

DATASET_DIR="/workspace/dataset/anal_motion"
OUTPUT_DIR="/workspace/ComfyUI/models/loras"
MODEL_PATH="/workspace/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
LORA_NAME="wan2.2_motion_anal_v1"

echo "=============================================================================="
echo "Starting Wan 2.2 Motion LoRA Training: ${LORA_NAME}"
echo "Dataset: ${DATASET_DIR}"
echo "Output:  ${OUTPUT_DIR}/${LORA_NAME}.safetensors"
echo "=============================================================================="

# Ensure directories exist
mkdir -p "${DATASET_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Step 1: Run caption generator
echo "[1/3] Generating text captions for video clips..."
python /workspace/story-video-engine/scripts/create_anal_captions.py "${DATASET_DIR}"

# Step 2: Navigate to training directory
cd /workspace/musubi-tuner

# Step 3: Launch training (750 steps for 10 clips)
echo "[2/3] Launching LoRA training (750 steps)..."
python train_network.py \
  --pretrained_model_name_or_path="${MODEL_PATH}" \
  --train_data_dir="${DATASET_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --output_name="${LORA_NAME}" \
  --network_module="networks.lora" \
  --network_dim=32 \
  --network_alpha=16 \
  --learning_rate=2e-4 \
  --max_train_steps=750 \
  --mixed_precision="bf16" \
  --gradient_checkpointing

echo "=============================================================================="
echo "[3/3] Training Complete!"
echo "LoRA saved to: ${OUTPUT_DIR}/${LORA_NAME}.safetensors"
echo "=============================================================================="

