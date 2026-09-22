from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path("/workspace/story-video-engine")

load_dotenv(BASE_DIR / ".env")

COMFY_DIR = Path("/workspace/ComfyUI")

OUTPUT_DIR = BASE_DIR / "output"
WORKFLOWS_DIR = BASE_DIR / "workflows"

COMFY_URL = os.getenv(
    "COMFY_URL",
    "http://127.0.0.1:8188",
)
COMFYUI_URL = COMFY_URL

STORY_PLANNER_MODEL = os.getenv(
    "STORY_PLANNER_MODEL",
    "gpt-5.6",
)

# Default production scene settings
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "1280"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "720"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "16"))
SCENE_DURATION = float(os.getenv("SCENE_DURATION", "5"))

QWEN_EDIT_WORKFLOW = (
    WORKFLOWS_DIR /
    "qwen_image_edit_2511_api.json"
)

WAN_I2V_WORKFLOW = (
    WORKFLOWS_DIR /
    "wan2.2_i2v_api.json"
)

WAN_T2V_WORKFLOW = (
    WORKFLOWS_DIR /
    "wan2.2_t2v_14b_api.json"
)
QWEN_IMAGE_WORKFLOW = WORKFLOWS_DIR / "qwen_image_2512_api.json"
JUGGERNAUT_XL_WORKFLOW = WORKFLOWS_DIR / "juggernaut_xl_t2i_api.json"

IMAGE_GENERATOR_MODEL = os.getenv("IMAGE_GENERATOR_MODEL", "juggernaut_xl_v9")
JUGGERNAUT_CHECKPOINT = os.getenv("JUGGERNAUT_CHECKPOINT", "Juggernaut-XL_v9.safetensors")
JUGGERNAUT_STEPS = int(os.getenv("JUGGERNAUT_STEPS", "35"))
JUGGERNAUT_CFG = float(os.getenv("JUGGERNAUT_CFG", "5.0"))
JUGGERNAUT_SAMPLER = os.getenv("JUGGERNAUT_SAMPLER", "dpmpp_2m")
JUGGERNAUT_SCHEDULER = os.getenv("JUGGERNAUT_SCHEDULER", "karras")

KEYFRAME_DENOISE = float(
    os.getenv("KEYFRAME_DENOISE", "0.70")
)
