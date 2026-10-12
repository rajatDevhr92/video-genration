import gc
import json
import os
import re
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .llm_provider import LLMProvider
from .gpu_memory import prepare_for_local_llm


class LocalLLMProvider(LLMProvider):

    def __init__(self, model_path=None):

        # Release ComfyUI VRAM BEFORE loading Qwen
        prepare_for_local_llm()

        target_path = model_path or os.getenv("LOCAL_PLANNER_MODEL") or "/workspace/models/qwen3-30b-a3b-instruct-2507"

        # DEEP SAFETY CHECK: If target_path looks like a local filesystem path but does NOT exist on disk, fall back to Qwen 7B
        is_local_path = target_path.startswith("/") or target_path.startswith(".") or "\\" in target_path or ":" in target_path
        if is_local_path and not os.path.exists(target_path):
            print(f"[LLM] Notice: Local model path '{target_path}' not found on disk. Falling back to 'Qwen/Qwen2.5-7B-Instruct'.")
            target_path = "Qwen/Qwen2.5-7B-Instruct"

        self.model_path = target_path

        self.max_new_tokens = int(
            os.getenv(
                "LOCAL_PLANNER_MAX_TOKENS",
                "32768",
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.model.eval()

    @staticmethod
    def _extract_json(text: str):

        text = text.strip()

        # Remove markdown fences.
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        text = text.strip()

        # --------------------------------------------------
        # Direct JSON
        # --------------------------------------------------

        try:
            return json.loads(text)

        except json.JSONDecodeError:
            pass

        # --------------------------------------------------
        # Find first JSON object
        # --------------------------------------------------

        start = text.find("{")

        if start == -1:
            raise ValueError(
                "No JSON object found in model output."
            )

        depth = 0
        in_string = False
        escape = False

        last_valid_end = None

        for i in range(start, len(text)):

            char = text[i]

            if in_string:

                if escape:
                    escape = False

                elif char == "\\":
                    escape = True

                elif char == '"':
                    in_string = False

                continue

            if char == '"':
                in_string = True

            elif char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:

                    candidate = text[
                        start:i + 1
                    ]

                    try:
                        return json.loads(candidate)

                    except json.JSONDecodeError:
                        last_valid_end = i

        # --------------------------------------------------
        # Give a useful truncation error
        # --------------------------------------------------

        if depth > 0:

            raise ValueError(
                "Model output appears to be truncated. "
                f"Unclosed JSON depth: {depth}. "
                f"Output length: {len(text)} characters."
            )

        raise ValueError(
            "Could not extract valid JSON from model output."
        )

    def generate_json(self, prompt: str):

        messages = [
            {
                "role": "user",
                "content": (
                    prompt
                    + "\n\n"
                    "FINAL OUTPUT RULE:\n"
                    "Return ONLY ONE valid JSON object.\n"
                    "Do not write explanations.\n"
                    "Do not write Markdown.\n"
                    "Do not use code fences.\n"
                    "Do not stop before the JSON object is complete."
                ),
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        )

        input_device = (
            self.model.get_input_embeddings()
            .weight.device
        )

        inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
        }

        input_length = inputs["input_ids"].shape[-1]

        print(
            f"[LLM] Prompt tokens: {input_length}"
        )

        print(
            f"[LLM] Max new tokens: "
            f"{self.max_new_tokens}"
        )

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id
                ),
            )

        generated_tokens = outputs[
            0
        ][
            input_length:
        ]

        result_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        print(
            f"[LLM] Generated tokens: "
            f"{len(generated_tokens)}"
        )

        print(
            f"[LLM] Generated characters: "
            f"{len(result_text)}"
        )

        # --------------------------------------------------
        # Always save raw output
        # --------------------------------------------------

        debug_path = "/workspace/local_llm_last_output.txt"

        with open(
            debug_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(result_text)

        print(
            f"[LLM] Raw output saved to: {debug_path}"
        )

        try:

            return self._extract_json(
                result_text
            )

        except ValueError as error:

            print(
                "[LLM] JSON extraction failed."
            )

            raise RuntimeError(
                "Local LLM returned invalid JSON.\n\n"
                f"Raw output saved to: {debug_path}\n\n"
                f"Parser error: {error}"
            ) from error

    def unload(self):

        print()
        print("=" * 70)
        print("LOCAL LLM MEMORY RELEASE")
        print("=" * 70)

        try:

            if hasattr(self, "model"):

                print(
                    "[LLM] Deleting local model..."
                )

                del self.model

        except Exception as error:

            print(
                f"[LLM] Model delete warning: {error}"
            )

        try:

            if hasattr(self, "tokenizer"):

                del self.tokenizer

        except Exception:
            pass

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            reserved = (
                torch.cuda.memory_reserved()
                / 1024**3
            )

            total = (
                torch.cuda.get_device_properties(0)
                .total_memory
                / 1024**3
            )

            print(
                "[MEMORY] After Local LLM unload: "
                f"allocated={allocated:.2f} GB, "
                f"reserved={reserved:.2f} GB, "
                f"total={total:.2f} GB"
            )

        print(
            "[LLM] Local model memory released."
        )