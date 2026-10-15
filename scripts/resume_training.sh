#!/usr/bin/env bash
set -e

DATASET_DIR="/workspace/dataset"
OUTPUT_DIR="/workspace/ComfyUI/models/loras"
AITOOLKIT_DIR="/workspace/ai-toolkit"
LORA_NAME="wan2.2_multi_motion_v1"
CHECKPOINT_500="/workspace/ComfyUI/models/loras/wan2.2_multi_motion_v1/wan2.2_multi_motion_v1_000000500.safetensors"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="0"

# Ensure dependencies are present
pip install python-dotenv || true

CONFIG_YAML="${AITOOLKIT_DIR}/config/wan_multi_motion_resume.yaml"

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
        # Resumes from your Step 500 saved weights:
        from_existing: "${CHECKPOINT_500}"
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
        lr: 1.5e-4
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

cd "${AITOOLKIT_DIR}"
python run.py "${CONFIG_YAML}"

