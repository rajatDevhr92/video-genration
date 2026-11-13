"""
Configuration constants for the video-generation and video-cloning project.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Core directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parent.parent))
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

OUTPUT_DIR = BASE_DIR / "output"
WORKFLOWS_DIR = BASE_DIR / "workflows"
COMFY_DIR = Path(os.getenv("COMFY_DIR", "/workspace/ComfyUI"))
COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188")
COMFYUI_URL = COMFY_URL

STORY_PLANNER_MODEL = os.getenv("STORY_PLANNER_MODEL", "gpt-5.6")

# ---------------------------------------------------------------------------
# Video defaults (used by WanVideoCloner and pipeline)
# ---------------------------------------------------------------------------
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "1280"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "720"))
VIDEO_FPS = float(os.getenv("VIDEO_FPS", "16"))
SCENE_DURATION = float(os.getenv("SCENE_DURATION", "5"))

# ---------------------------------------------------------------------------
# Cloning Backend Selection
# ---------------------------------------------------------------------------
# "fun_control" - new default, uses the Wan 2.2 14B Fun Control workflow.
# "legacy_i2v"  - keep the old I2V/V2V flow for backward compatibility.
CLONING_BACKEND = os.getenv("CLONING_BACKEND", "fun_control")

# ---------------------------------------------------------------------------
# Wan 2.2 Fun Control Workflow & Model Configuration
# ---------------------------------------------------------------------------
FUN_CONTROL_WORKFLOW = WORKFLOWS_DIR / "wan2.2_fun_control_api.json"

# Model filenames referenced inside the Fun Control workflow.
FUN_CONTROL_CLIP_MODEL = os.getenv(
    "FUN_CONTROL_CLIP_MODEL",
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
)
FUN_CONTROL_VAE_MODEL = os.getenv(
    "FUN_CONTROL_VAE_MODEL",
    "wan_2.1_vae.safetensors",
)
FUN_CONTROL_HIGH_NOISE_MODEL = os.getenv(
    "FUN_CONTROL_HIGH_NOISE_MODEL",
    "split_files/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors",
)
FUN_CONTROL_LOW_NOISE_MODEL = os.getenv(
    "FUN_CONTROL_LOW_NOISE_MODEL",
    "split_files/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors",
)
FUN_CONTROL_LORA_HIGH = os.getenv(
    "FUN_CONTROL_LORA_HIGH",
    "split_files/loras/wan2.2_fun_control_lightx2v_4steps_lora_v1_high_noise.safetensors",
)
FUN_CONTROL_LORA_LOW = os.getenv(
    "FUN_CONTROL_LORA_LOW",
    "split_files/loras/wan2.2_fun_control_lightx2v_4steps_lora_v1_low_noise.safetensors",
)

# Diffusion parameters for Fun Control.
# Default: 20 steps and no LoRA.
USE_LIGHTNING_LORA = os.getenv("USE_LIGHTNING_LORA", "False").lower() in ("1", "true", "yes")
FUN_CONTROL_STEPS = int(os.getenv("FUN_CONTROL_STEPS", "20"))
FUN_CONTROL_STEPS_LORA = int(os.getenv("FUN_CONTROL_STEPS_LORA", "4"))
FUN_CONTROL_CFG = float(os.getenv("FUN_CONTROL_CFG", "2.0"))
FUN_CONTROL_SAMPLER = os.getenv("FUN_CONTROL_SAMPLER", "euler")
FUN_CONTROL_SCHEDULER = os.getenv("FUN_CONTROL_SCHEDULER", "simple")
FUN_CONTROL_SHIFT = float(os.getenv("FUN_CONTROL_SHIFT", "5.0"))

# Control video generation mode: "pose+depth", "pose", "depth" (default), "canny"
CONTROL_VIDEO_MODE = os.getenv("CONTROL_VIDEO_MODE", "depth")

# Multi-chunk slicing settings: 10.0s at 16 FPS = 161 frames (4n + 1 required by Wan 2.2)
CHUNK_DURATION_SEC = float(os.getenv("CHUNK_DURATION_SEC", "10.0"))
CHUNK_TARGET_FRAMES = int(os.getenv("CHUNK_TARGET_FRAMES", "161"))

# Re-anchor interval for chunk chaining (prevents color and detail drift)
REANCHOR_EVERY_N_CHUNKS = int(os.getenv("REANCHOR_EVERY_N_CHUNKS", "15"))

# ---------------------------------------------------------------------------
# Quality gate thresholds and retry settings
# ---------------------------------------------------------------------------
QUALITY_MOTION_THRESHOLD = float(os.getenv("QUALITY_MOTION_THRESHOLD", "0.35"))
QUALITY_FLICKER_THRESHOLD = float(os.getenv("QUALITY_FLICKER_THRESHOLD", "35.0"))
QUALITY_MAX_RETRIES = int(os.getenv("QUALITY_MAX_RETRIES", "1"))
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

# Node IDs inside wan2.2_fun_control_api.json
FUN_CONTROL_NODE_IDS = {
    "clip": "1",
    "vae": "2",
    "unet_high": "3",
    "unet_low": "4",
    "lora_high": "5",
    "lora_low": "6",
    "shift_high": "7",
    "shift_low": "8",
    "prompt_pos": "9",
    "prompt_neg": "10",
    "fun_control": "11",
    "sampler_high": "12",
    "sampler_low": "13",
    "vae_decode": "14",
    "create_video": "15",
    "save_video": "16",
    "load_control_video": "20",
    "get_video_components": "21",
    "load_ref_image": "1000",
}

# ---------------------------------------------------------------------------
# Default Pipeline Prompts (Qwen 2D Anime Redraw & Wan 2.2 Fun Control)
# ---------------------------------------------------------------------------
QWEN_KEYFRAME_PROMPT = (
    "Masterpiece 2D Japanese anime illustration, Kyoto Animation and Makoto Shinkai style.\n"
    "MANDATORY 100% FULL-FRAME 2D ANIME REDRAW: The ENTIRE image in Image 1 must be completely transformed into a cohesive, hand-drawn 2D anime illustration. Every single square inch — all characters, full bodies, furniture, sofa, couch, walls, doors, cabinets, and room background — must be rendered in pure 2D cel-shaded anime style with clean anime ink outlines. ABSOLUTELY ZERO live-action photograph elements or real human textures.\n\n"
    "1. ENVIRONMENT & FURNITURE - 100% 2D ANIME ART:\n"
    "   - Completely repaint the sofa, couch, cushions, bed, seating, and floor into stylized 2D anime furniture with clean anime outline strokes, smooth anime shading, and vibrant anime colors. Eliminate all photographic leather, cloth grain, and realistic textures.\n"
    "   - Transform the entire background (walls, doors, kitchen cabinets, countertops, ceiling, lighting) into hand-painted 2D Japanese anime background art with luminous anime lighting and rich cinematic color palette.\n"
    "   - Remove all text, logos, stamps, and watermarks.\n\n"
    "2. FEMALE CHARACTER - FULL-BODY 2D ANIME CONVERSION ({f_slot_str}):\n"
    "   - Identify the female actor in Image 1.\n"
    "   - Completely transform her into the gorgeous 2D anime woman strictly matching {f_slot_str}.\n"
    "   - Render her stylized anime hairstyle and hair color from {f_slot_str}, large expressive detailed anime eyes, and clean anime facial contours.\n"
    "   - Render her ENTIRE body from head to toe — face, neck, shoulders, chest, arms, hands, back, hips, thighs, knees, legs, and feet — in smooth 2D cel-shaded anime skin with clean dark anime outline linework, radiant anime skin tone, and soft anime shading.\n\n"
    "3. MALE CHARACTER - FULL-BODY 2D ANIME CONVERSION ({m_slot_str}):\n"
    "   - Identify the male actor in Image 1.\n"
    "   - Completely transform him into the handsome 2D anime man strictly matching {m_slot_str}.\n"
    "   - Render his stylized anime haircut from {m_slot_str}, sharp defined anime jawline, and expressive anime eyes.\n"
    "   - Render his ENTIRE body from head to toe — face, neck, shoulders, chest, torso, abdomen, hips, groin, private areas, thighs, knees, legs, and feet — in pure 2D anime style with clean anime ink outlines and smooth flat cel-shaded anime skin.\n"
    "   - COMPLETELY ELIMINATE all realistic human body hair, skin pores, photographic blemishes, and live-action skin textures across his entire lower body, thighs, and legs.\n\n"
    "4. POSE, ANGLE & INTERACTION LOCK:\n"
    "   - Faithfully preserve 100% of the original physical body positions, intimate interaction, limb angles, spatial interaction, and camera perspective from Image 1.\n"
    "   - The entire image must be a unified 2D anime animation still frame. High-end anime aesthetic, crisp cel-shading, pure 2D anime art, no photorealism, no real human skin, no photograph."
)

QWEN_KEYFRAME_NEGATIVE = (
    "photorealistic, realistic, photograph, raw photo, live action, live action footage, "
    "real human skin, skin pores, natural skin texture, body hair, photographic texture, "
    "real leather, photographic furniture, real room photo, real couch, real sofa, "
    "camera noise, 3d, 3d render, cgi, realistic lighting, bad anatomy, extra limbs, "
    "fused bodies, text, watermark, logo, Princess CUM, subtitles"
)

FUN_CONTROL_POSITIVE_PROMPT = (
    "masterpiece 2D Japanese anime animation, Kyoto Animation and Makoto Shinkai aesthetic, "
    "full-frame 2D anime scene in motion, hand-drawn anime characters and hand-painted anime background, "
    "smooth fluid hand-drawn anime movement, clean cel-shaded anime linework, gorgeous luminous anime lighting, "
    "vibrant anime color palette, pure 2D anime cinematography, high quality anime film, no live action"
)

FUN_CONTROL_NEGATIVE_PROMPT = (
    "photorealistic, realistic, photograph, raw photo, live action, live action footage, real human, "
    "real human skin, skin pores, body hair, photographic furniture, real room photo, real couch, real sofa, "
    "3d, 3d render, camera noise, real life footage, "
    "色调艳丽，过曝，静态，细节模糊不清，字幕，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合"
)

# ---------------------------------------------------------------------------
# Existing Wan Workflows (kept for legacy_i2v mode & other generators)
# ---------------------------------------------------------------------------
WAN_I2V_WORKFLOW = WORKFLOWS_DIR / "wan2.2_i2v_api.json"
WAN_V2V_WORKFLOW = WORKFLOWS_DIR / "wan2.2_v2v_api.json"
WAN_T2V_WORKFLOW = WORKFLOWS_DIR / "wan2.2_t2v_14b_api.json"

# ---------------------------------------------------------------------------
# Image generation & editing workflows
# ---------------------------------------------------------------------------
QWEN_EDIT_WORKFLOW = WORKFLOWS_DIR / "qwen_image_edit_2511_api.json"
QWEN_IMAGE_WORKFLOW = WORKFLOWS_DIR / "qwen_image_2512_api.json"
JUGGERNAUT_XL_WORKFLOW = WORKFLOWS_DIR / "juggernaut_xl_t2i_api.json"

IMAGE_GENERATOR_MODEL = os.getenv("IMAGE_GENERATOR_MODEL", "juggernaut_xl_v9")
_env_ckpt = os.getenv("JUGGERNAUT_CHECKPOINT", "Juggernaut-XL_v9.safetensors")
if "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" in _env_ckpt:
    JUGGERNAUT_CHECKPOINT = "Juggernaut-XL_v9.safetensors"
else:
    JUGGERNAUT_CHECKPOINT = _env_ckpt

JUGGERNAUT_STEPS = int(os.getenv("JUGGERNAUT_STEPS", "35"))
JUGGERNAUT_CFG = float(os.getenv("JUGGERNAUT_CFG", "5.0"))
JUGGERNAUT_SAMPLER = os.getenv("JUGGERNAUT_SAMPLER", "dpmpp_2m")
JUGGERNAUT_SCHEDULER = os.getenv("JUGGERNAUT_SCHEDULER", "karras")
KEYFRAME_DENOISE = float(os.getenv("KEYFRAME_DENOISE", "0.70"))
