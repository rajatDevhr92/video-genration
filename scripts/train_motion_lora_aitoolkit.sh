#!/usr/bin/env bash
# ==============================================================================
# Wan 2.2 Multi-Category Motion LoRA Training Script (AI-Toolkit Implementation)
# ==============================================================================

set -e

DATASET_DIR="/workspace/dataset"
OUTPUT_DIR="/workspace/ComfyUI/models/loras"
AITOOLKIT_DIR="/workspace/ai-toolkit"
LORA_NAME="wan2.2_multi_motion_v1"

echo "=============================================================================="
echo "Starting Wan 2.2 Multi-Category Motion LoRA Training: ${LORA_NAME}"
echo "Dataset: ${DATASET_DIR}"
echo "Output:  ${OUTPUT_DIR}/${LORA_NAME}.safetensors"
echo "=============================================================================="

# Ensure directories exist
mkdir -p "${DATASET_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Step 1: Run Master Multi-Category Caption Generator
echo "[1/4] Generating structured text captions across all dataset subfolders..."
python /workspace/story-video-engine/scripts/generate_all_captions.py "${DATASET_DIR}"

# Step 2: Clone and set up AI-Toolkit if not present
if [ ! -d "${AITOOLKIT_DIR}" ]; then
    echo "[2/4] Cloning ai-toolkit repository to ${AITOOLKIT_DIR}..."
    git clone https://github.com/ostris/ai-toolkit.git "${AITOOLKIT_DIR}"
    cd "${AITOOLKIT_DIR}"
    git submodule update --init --recursive
fi

echo "[2.5/4] Synchronizing AI-Toolkit dependencies..."
cd "${AITOOLKIT_DIR}"
pip install -r requirements.txt || true

# Step 3: Create YAML configuration file for Wan 2.2 Multi-Category LoRA training
CONFIG_YAML="${AITOOLKIT_DIR}/config/wan_multi_motion.yaml"
mkdir -p "${AITOOLKIT_DIR}/config"

echo "[3/4] Creating AI-Toolkit YAML configuration at ${CONFIG_YAML}..."
cat << EOF > "${CONFIG_YAML}"
job: extension
config:
  name: "${LORA_NAME}"
  process:
    - type: sd_trainer
      training_folder: "${OUTPUT_DIR}"
      device: cuda:0
      trigger_word: "motion_lora"
      network:
        type: lora
        linear: 32
        linear_alpha: 16
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
      logging:
        log_every: 10
      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 2
        train_unet: true
        gradient_checkpointing: true
        optimizer: adamw8bit
        lr: 2e-4
        dtype: bf16
      model:
        name_or_path: "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        arch: "wan21:14b"
        quantize: true
      datasets:
        - folder_path: "${DATASET_DIR}"
          caption_ext: "txt"
          resolution: [832, 480]
          num_frames: 49
          auto_frame_count: true
          exts: ["mp4"]
          cache_latents_to_disk: true
meta:
  name: "${LORA_NAME}"
  version: "1.0"
EOF

# Step 4: Launch Training
echo "[4/4] Launching AI-Toolkit Wan 2.2 Multi-Category LoRA training..."
cd "${AITOOLKIT_DIR}"
python run.py "${CONFIG_YAML}"

echo "=============================================================================="
echo "Training Complete!"
echo "LoRA saved to: ${OUTPUT_DIR}/${LORA_NAME}.safetensors"
echo "=============================================================================="

