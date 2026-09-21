import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path("/workspace/story-video-engine")

load_dotenv(PROJECT_ROOT / ".env")


__all__ = [
    "AudioGenerator",
    "generate_story_audio",
    "resolve_character_voice",
]


FEMALE_VOICES = ["nova", "coral", "shimmer"]
MALE_VOICES = ["onyx", "echo", "ash"]

DEFAULT_NARRATOR_VOICE = "alloy"


def resolve_character_voice(
    character_id: str,
    character_obj: dict = None,
    requested_voice: str = None,
) -> str:
    """
    Dynamically select and validate an appropriate TTS voice based on character gender / description.
    Prevents assigning male voices (e.g. onyx) to female characters (e.g. Elena/Sarah), and vice versa.
    """
    gender = None
    if character_obj:
        appearance = str(character_obj.get("appearance", "")).lower()
        anchor = str(character_obj.get("visual_anchor", "")).lower()
        name = str(character_obj.get("name", "")).lower()
        full_text = f"{name} {appearance} {anchor}"

        if any(w in full_text for w in ["female", "woman", "girl", "lady", "she", "her"]):
            gender = "female"
        elif any(w in full_text for w in ["male", "man", "boy", "guy", "he", "his"]):
            gender = "male"

    if requested_voice and requested_voice.lower() != "alloy":
        req_lower = requested_voice.lower()
        if gender == "female" and req_lower in MALE_VOICES:
            return "nova"
        if gender == "male" and req_lower in FEMALE_VOICES:
            return "onyx"
        return requested_voice

    if gender == "female":
        return "nova"
    if gender == "male":
        return "onyx"

    # Fallback by character index: odd/1 -> nova (female default lead), 2 -> onyx (male)
    if "001" in character_id or character_id.endswith("_1"):
        return "nova"
    elif "002" in character_id or character_id.endswith("_2"):
        return "onyx"
    elif "003" in character_id:
        return "coral"
    else:
        return "echo"



class AudioGenerator:

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model = os.getenv(
            "STORY_TTS_MODEL",
            "gpt-4o-mini-tts",
        )

    @staticmethod
    def run(command):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr
            )

        return result

    def generate_segment(
        self,
        text,
        voice,
        output_path,
        instructions=None,
    ):
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        request = {
            "model": self.model,
            "voice": voice,
            "input": text,
            "response_format": "wav",
        }

        if instructions:
            request["instructions"] = instructions

        with self.client.audio.speech.with_streaming_response.create(
            **request
        ) as response:
            response.stream_to_file(
                output_path
            )

        return output_path

    @staticmethod
    def is_refusal_text(text: str) -> bool:
        refusal_phrases = [
            "i'm sorry", "i cannot", "i can't assist", "as an ai",
            "cannot fulfill", "safety guidelines", "policy", "cannot generate"
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in refusal_phrases)

    def create_scene_audio(
        self,
        audio_plan,
        output_path,
        duration=5.1,
        character_map=None,
    ):
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        narration = audio_plan.get(
            "narration",
            [],
        )

        dialogue = audio_plan.get(
            "dialogue",
            [],
        )

        segments = []

        with tempfile.TemporaryDirectory(
            prefix="story_audio_"
        ) as temp_dir:

            temp_dir = Path(temp_dir)

            index = 0

            # Narration
            for item in narration:
                text = str(
                    item.get("text", "")
                ).strip()

                if not text or self.is_refusal_text(text):
                    continue

                voice = item.get(
                    "voice",
                    DEFAULT_NARRATOR_VOICE,
                )

                segment = (
                    temp_dir
                    / f"{index:03d}.wav"
                )

                self.generate_segment(
                    text=text,
                    voice=voice,
                    output_path=segment,
                    instructions=(
                        "Cinematic narration. "
                        "Natural, restrained and serious. "
                        "Speak clearly and avoid overacting."
                    ),
                )

                segments.append(segment)
                index += 1

            # Dialogue
            for item in dialogue:
                text = str(
                    item.get("text", "")
                ).strip()

                if not text or self.is_refusal_text(text):
                    continue

                character_id = item.get(
                    "character_id",
                    "",
                )

                char_obj = (character_map or {}).get(character_id)
                requested_voice = item.get("voice")

                voice = resolve_character_voice(
                    character_id=character_id,
                    character_obj=char_obj,
                    requested_voice=requested_voice,
                )

                segment = (
                    temp_dir
                    / f"{index:03d}.wav"
                )

                self.generate_segment(
                    text=text,
                    voice=voice,
                    output_path=segment,
                    instructions=(
                        "Natural cinematic dialogue. "
                        "Emotionally grounded and restrained. "
                        "Do not overact."
                    ),
                )

                segments.append(segment)
                index += 1

            # No spoken audio:
            # create a silent track so every scene has
            # a predictable audio stream.
            if not segments:
                self.run([
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=48000:cl=stereo",
                    "-t",
                    str(duration),
                    "-c:a",
                    "pcm_s16le",
                    str(output_path),
                ])

                return output_path

            # Concatenate speech segments.
            concat_file = (
                temp_dir / "concat.txt"
            )

            with open(
                concat_file,
                "w",
                encoding="utf-8",
            ) as f:
                for segment in segments:
                    f.write(
                        f"file '{segment.resolve()}'\n"
                    )

            self.run([
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ])

        return output_path

    def mux_video_audio(
        self,
        video_path,
        audio_path,
        output_path,
    ):
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.run([
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ])

        return output_path


def generate_story_audio(job_dir, start_scene=1, end_scene=None):
    """
    Generate audio for story scenes in job_dir and mux them into video files.
    """
    import json
    job_dir = Path(job_dir)
    plan_path = job_dir / "story_plan.json"

    if not plan_path.exists():
        raise FileNotFoundError(f"story_plan.json not found: {plan_path}")

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    scenes = plan.get("scenes", [])
    if not scenes:
        raise ValueError("No scenes found in story_plan.json")

    character_map = {
        c["id"]: c for c in plan.get("characters", [])
    }

    if end_scene is None:
        end_scene = len(scenes)

    selected_scenes = [
        s for s in scenes
        if start_scene <= s.get("order", 0) <= end_scene
    ]

    generator = AudioGenerator()
    results = []

    for scene in selected_scenes:
        scene_id = scene["id"]
        scene_dir = job_dir / "scenes" / scene_id

        video = scene_dir / "video.mp4"
        audio = scene_dir / "audio.wav"
        video_audio = scene_dir / "video_with_audio.mp4"

        audio_plan = scene.get("audio", {})
        if not isinstance(audio_plan, dict):
            audio_plan = {}

        generator.create_scene_audio(
            audio_plan=audio_plan,
            output_path=audio,
            duration=scene.get("duration", 5.1),
            character_map=character_map,
        )

        if video.exists():
            generator.mux_video_audio(
                video_path=video,
                audio_path=audio,
                output_path=video_audio,
            )

        metadata_path = scene_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            metadata["audio"] = str(audio)
            metadata["video_with_audio"] = str(video_audio)

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        results.append({
            "scene_id": scene_id,
            "audio": str(audio),
            "video_with_audio": str(video_audio) if video.exists() else None,
        })

    return results
