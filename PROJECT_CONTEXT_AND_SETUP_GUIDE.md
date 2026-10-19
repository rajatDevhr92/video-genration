# Story Video Engine: Complete Project Context, Architecture & Mac M4 Setup Guide

This document captures the complete architectural knowledge, implemented fixes, identified issues, and step-by-step local Mac mini M4 setup for the **Story Video Engine** project. Any developer or AI agent reading this document has full context on the codebase history, current pipeline behavior, and future deployment roadmap.

---

## 1. System Architecture Overview

The **Story Video Engine** is a multi-stage cinematic video generation pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. STORY & SCENE PLANNING                                                   │
│    - app/story_planner.py & app/story_parser.py                             │
│    - LLM creates structured 15-scene story_plan.json                         │
│    - app/compiler.py maps 3D turnaround camera angles & wardrobe tracking  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. CHARACTER REFERENCES & KEYFRAME COMPOSITION                              │
│    - app/character_reference_generator.py (Krea 2 / SDXL Turnarounds)        │
│    - app/composition_generator.py (Scene 1 initial camera framing)          │
│    - app/keyframe_generator.py (Qwen Image Edit Plus Stage A & Stage B)      │
│    - app/continuity.py (Deterministic state tracking: clothing, nudity)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. VIDEO MOTION GENERATION                                                  │
│    - app/video_generator.py (Wan 2.2 / 2.1 I2V via ComfyUI)                 │
│    - Motion LoRAs (LightX2V 4-step distilled LoRA, action trigger LoRAs)     │
│    - Last-frame extraction for seamless scene-to-scene transitions          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. AUDIO GENERATION & MASTER ASSEMBLY                                       │
│    - add_audio_to_story.py & app/audio_generator.py (Dialogue TTS & Foley)  │
│    - assemble_story.py & app/video_assembler.py (FFmpeg concat stream copy) │
│    - Output: output/<job_id>/final/final_with_audio.mp4                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Issues Found & Solved

### Issue 1: `AttributeError: 'str' object has no attribute 'get'` in Continuity Tracker
- **Symptoms:** Generating a story plan halted at Scene 4 with:
  ```text
  File "/workspace/story-video-engine/app/continuity.py", line 88, in advance
    cs = self.state.get(ev.get("character_id"))
  AttributeError: 'str' object has no attribute 'get'
  ```
- **Root Cause:** In LLM-crafted story plans, `wardrobe_events` was formatted as a list of plain strings (e.g. `["remove_top"]` or `["remove_bottom"]`) rather than structured objects (`[{"character_id": "character_001", "event": "remove_top"}]`).
- **Solution Implemented in [`app/continuity.py`](file:///d:/rajat/story-video-engine/app/continuity.py):**
  - Added `_normalize_events()` to `ContinuityTracker`:
    - Handles strings (`"remove_top"`, `"Sunny: remove_top"`, `"remove_all"`).
    - Maps character names to IDs.
    - If the character ID is missing, disambiguates the target character using scene text, possessives (`"Sunny's blouse"`), gender pronouns, and current clothing status.
    - Normalizes `scene["wardrobe_events"]` in-place so downstream code always receives clean dictionaries.
  - Guarded [`app/compiler.py`](file:///d:/rajat/story-video-engine/app/compiler.py) to check `isinstance(e, dict)` before calling `e.get()`.
  - Added unit test in [`test_local_validation.py`](file:///d:/rajat/story-video-engine/test_local_validation.py).

### Issue 2: Resume Capability When Resuming from Middle Scenes
- **Symptoms:** Resuming a failed or paused job using `--start-scene 4` restarted keyframe generation from scratch (mode `t2i`) rather than using Scene 3's last frame.
- **Solution Implemented in [`app/scene_manager.py`](file:///d:/rajat/story-video-engine/app/scene_manager.py):**
  - When `start_scene > 1`, `SceneManager` now automatically checks `job_dir / "scenes" / prev_scene_id / "last_frame.png"` on disk. If found, it initializes `previous_last_frame`, allowing crop-zoom and continuation denoise levels ($0.30$–$0.35$).

### Issue 3: Missing CLI Script for Master Video Assembly
- **Symptoms:** After running `add_audio_to_story.py`, there was no standalone CLI command to concatenate all scene clips into the final movie outside the API server.
- **Solution Implemented:**
  - Created [`assemble_story.py`](file:///d:/rajat/story-video-engine/assemble_story.py). Can be executed directly:
    ```bash
    python assemble_story.py output/<job_id>
    ```

### Issue 4: Direct Story Plan API Route
- **Feature Implemented in [`api_server.py`](file:///d:/rajat/story-video-engine/api_server.py):**
  - Added `POST /api/v1/stories/plan` and `POST /api/v1/stories/from-plan`.
  - Accepts a full `story_plan.json` directly, skipping the LLM story generator and going straight into reference, keyframe, and video generation.

---

## 3. Discovered Visual Consistency Problems & Architectural Solution

### The Problem (Frame Discontinuity)
When comparing early frames against later scene frames:
1. **Character Identity Drift:** Characters morph (e.g. East Asian woman `Sunny` with straight jet-black hair in Scene 1 turns into a blonde Caucasian woman in later close-up scenes).
2. **Background Loss:** The detailed bedroom (wooden bed, white duvet, bedside table, mirror, window) dissolves into a generic flat peach/orange wall.
3. **Lighting Shift:** Scene 1 renders in natural daylight (~5000K), while later scenes shift into an extreme orange/sepia tint (~2700K).

### Root Causes
1. **No Environment Image Reference:** Characters have 3-panel turnaround reference sheets, but locations have **zero** image references. The model hallucinates the room from scratch whenever camera angles or framing change.
2. **Denoise Too High on Transition ($0.85$):** When undressing or changing poses, high denoise overrides the base image, causing diffusion defaults (blonde Caucasian bias) to take over.
3. **Subjective Lighting Terminology:** Prompts like `"Soft warm indoor lighting"` are interpreted randomly per seed (from clean daylight to heavy monochrome amber).

### The 4-Pillar Architectural Solution
1. **Master Location Plate:** Save Scene 1's wide composition as `master_location_plate.png` and feed it as the background anchor for subsequent scenes.
2. **Physical Identity Injection:** Prepend exact physical descriptions in all prompt stages (`Sunny: 22yo East Asian woman, jet-black straight hair, dark brown eyes`) and add negative locks (`(blonde hair, fair hair, Caucasian features:1.4)`).
3. **Calibrated Lighting Directives:** Replace `"warm lighting"` with `5000K neutral daylight through window, natural fill, realistic skin tones` and negative prompt `(heavy orange tint, sepia, amber wash:1.4)`.
4. **Reinhard Color Matching:** Apply a post-keyframe color transfer step matching the histogram and luminance of all new keyframes to Scene 1 before feeding into Wan 2.2.

---

## 4. Mac mini M4 Hardware & Model Sizing Guide

### The Memory Reality on Apple Silicon (16 GB vs 24 GB+ RAM)
- **16 GB Mac mini M4:**
  - Unified Memory is shared between macOS, CPU, and GPU.
  - **Wan 2.2 14B model cannot run:** 14B DiT + UMT5-XXL + VAE + video latents requires **22 GB–28 GB working VRAM**. Running it on a 16 GB Mac will cause immediate Out-Of-Memory (OOM) crashes or system freezes.
  - **The Solution for 16 GB M4:** Use **Wan 2.1 1.3B (GGUF Q4 or FP8)**.

| Component | Model on Cloud Server | Model on 16 GB Mac mini M4 | Memory on Mac |
| :--- | :--- | :--- | :--- |
| **Story Planning** | `gpt-5.6` | `gpt-5.6` or local Qwen 7B | ~300 MB |
| **Character T2I** | Krea 2 T2I / SDXL | SDXL Juggernaut XL | ~6 GB |
| **Keyframe Refinement**| Qwen Image Edit Plus | Qwen Edit / SDXL Inpaint | ~7 GB |
| **Video Generation** | **Wan 2.2 14B FP8** | **Wan 2.1 1.3B (GGUF Q4)** | **~8 GB – 10 GB** |
| **Audio (TTS)** | OpenAI TTS | OpenAI TTS / Kokoro TTS | ~500 MB |
| **Assembly** | FFmpeg | FFmpeg | ~150 MB |

### What is AI Upscaling (480P $\to$ 1080P)?
- **Wan 1.3B generates at 480P (832×480)** to stay under 10 GB RAM.
- An **AI Upscaler** (such as `RealESRGAN_x4plus` or `4x-UltraSharp`) is a lightweight neural network (~60 MB) that adds sharp skin textures, hair strands, and eye details to reconstruct the 480p frames into **crystal-clear 1080p** in ~5–10 seconds.

---

## 5. Step-by-Step Setup on Mac mini M4

### Step 1: Install System Prerequisites
```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Git, Python 3.11, and FFmpeg
brew install git python@3.11 ffmpeg
```

### Step 2: Install ComfyUI with Metal (MPS) Acceleration
```bash
mkdir -p ~/ai-video
cd ~/ai-video
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

python3.11 -m venv venv
source venv/bin/activate

# Install Apple Silicon native PyTorch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip install -r requirements.txt

# Install Custom Nodes
cd custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF.git
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
cd ..
```

### Step 3: Download Models into ComfyUI
Place the downloaded models in `~/ai-video/ComfyUI/models/`:

1. **Diffusion Model (Wan 1.3B GGUF):**
   - Download `Wan2.1-I2V-1.3B_Q4_K_M.gguf` from HuggingFace (`city96/Wan2.1-I2V-1.3B-GGUF`).
   - Place in: `models/diffusion_models/` (or `models/unet/`).
2. **Text Encoder:**
   - Download `umt5_xxl_fp8_e4m3fn_scaled.safetensors`.
   - Place in: `models/text_encoders/`.
3. **Image Encoder (CLIP Vision):**
   - Download `clip_vision_h.safetensors`.
   - Place in: `models/clip_vision/`.
4. **Video VAE:**
   - Download `wan_2.1_vae.safetensors`.
   - Place in: `models/vae/`.
5. **Character / Keyframe Checkpoint:**
   - Download `Juggernaut-XL_v9.safetensors`.
   - Place in: `models/checkpoints/`.
6. **AI Upscaler:**
   - Download `4x-UltraSharp.pth`.
   - Place in: `models/upscale_models/`.

### Step 4: Configure the Story Video Engine
```bash
cd ~/ai-video
git clone <your-repo-url> story-video-engine
cd story-video-engine

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create/Edit `.env`:
```ini
COMFY_URL=http://127.0.0.1:8188
VIDEO_WIDTH=832
VIDEO_HEIGHT=480
VIDEO_FPS=16
SCENE_DURATION=5
OPENAI_API_KEY=your_key_here
STORY_PLANNER_MODEL=gpt-5.6
```

### Step 5: Start Generation
- **Terminal 1 (ComfyUI Backend):**
  ```bash
  cd ~/ai-video/ComfyUI
  source venv/bin/activate
  python main.py
  ```
- **Terminal 2 (Engine API Server):**
  ```bash
  cd ~/ai-video/story-video-engine
  source venv/bin/activate
  python api_server.py
  ```

Submit your story plan:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/stories/plan \
  -H "Content-Type: application/json" \
  -d @output/story_c5e40af94e53/story_plan.json
```

---

## 6. Recommended Operational Mode: The Hybrid Workflow

- **Local Mac mini M4:** Use for fast script iterations, story planning, dialogue TTS, and Wan 1.3B draft previews (~2 min/scene, 100% free).
- **Cloud Server (RunPod / Vast.ai with RTX 4090):** Use for final 720p master renders with the full **Wan 2.2 14B model + 4-step distilled LoRA** by submitting the validated `story_plan.json` to the cloud API.

