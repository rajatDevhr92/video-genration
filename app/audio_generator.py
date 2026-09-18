import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path("/workspace/story-video-engine")

load_dotenv(PROJECT_ROOT / ".env")


CHARACTER_VOICES = {
    "character_001": "onyx",
    "character_002": "coral",
    "character_003": "echo",
    "character_004": "nova",
}

DEFAULT_NARRATOR_VOICE = "alloy"


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

                voice = item.get(
                    "voice"
                ) or CHARACTER_VOICES.get(
                    character_id,
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