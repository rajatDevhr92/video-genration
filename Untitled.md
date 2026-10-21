# Story-to-Video Pipeline Review

Your pipeline design is sound, and the new modular planner plus continuity tracking fixed the biggest problem (clothing and gender mix-ups). What still holds it back is not the models. It is a handful of small code gaps that quietly undo the consistency work, plus two slow or mismatched model settings. Fixing about ten small items will help more than swapping any model.

## How the pipeline works today

1. **Story in.** Story text is split into scenes of about 5 seconds.
2. **Planner.** A local Qwen3-30B (or OpenAI) lists the cast, then fills in a fixed set of facts per scene. A compiler turns those facts into prompts.
3. **Continuity.** A tracker remembers, per character, what is worn and what was removed, so clothing cannot flip between scenes.
4. **Character sheets.** Krea 2 draws one reference sheet per character.
5. **Keyframes.** Qwen-Image-Edit-2511 draws the first frame of each scene from the character sheets, chained from the previous scene's last frame.
6. **Video.** Wan 2.2 image-to-video animates each keyframe (81 frames at 16 fps, about 5 seconds).
7. **Audio and assembly.** OpenAI TTS makes the voice, ffmpeg joins every scene into the final video.

## Health check, stage by stage

| Stage | Health | Biggest weakness | Action |
| --- | --- | --- | --- |
| Planner | Good | Silent fallbacks hide planner failures; every scene is forced into one location | Fix |
| Continuity | Good | Keyword fallback in the anatomy check can fire wrongly | Fix |
| Character sheets (Krea 2) | Good | Negative prompt does nothing at cfg 1.0 | Tune |
| Keyframes (Qwen Edit) | Fair | 40 steps with no speed-up; only one reference image slot; no identity check yet | Fix |
| Video (Wan 2.2) | Fair | Motion LoRA trained on a different Wan version and attached to one of the two experts | Fix |
| Audio | Weak | Gender guess matches letters inside words ("she" in "shelf") | Fix |
| Assembly | Good | Scene length mismatch can drift audio | Tune |
| Memory (GPU) | Fair | Local LLM is not always unloaded before ComfyUI starts | Fix |

## Where the pipeline is weak

### 1. Planner and prompts

- **Everything happens in one place.** The plan hard-codes `location_001` and one lighting line ("soft warm indoor") for every scene. A story that moves from a street to a bedroom to a forest still renders as the same room.
- **Silent fallbacks.** When the LLM output is bad, the code quietly uses defaults. You get a video, but not the story you wrote, and nothing tells you.
- **Prompt pollution.** Long free text from the planner is pasted next to the structured facts, so old words can re-introduce clothing or body details the tracker removed.
- **Fix:** log and surface every fallback, let each scene carry its own location and lighting, and build prompts only from the closed facts.

### 2. Continuity tracking

- The fallback that scans free text for words like "penetrat" or "oral" matches parts of other words and can trigger the wrong image pass. Use planner-emitted closed acts only, and drop the keyword fallback.
- Gender matching is already word-boundary based here, which is good. Keep it as the single source of truth and reuse it in audio.

### 3. Character sheets and identity

- Krea 2 runs at cfg 1.0, so its negative prompt ("no collage, no split screen") has no effect. Rely on a clear positive prompt instead.
- Identity checking (`identity_check.py`) is written but not wired in. This is the cheapest big win: it catches a wrong face and retries with a new seed.
- Seeds use Python `hash()`, which changes every run. The same character gets a different seed after a restart. Use a stable hash (for example MD5 of the character id).

### 4. Keyframes

- Qwen Edit runs 40 steps at cfg 4.0 with no speed-up LoRA. It is the slowest image step. A 4- or 8-step Lightning LoRA for Qwen-Image-Edit would cut keyframe time several times with small quality loss. Test it on a few scenes first.
- The workflow file has only image1, but keyframe\_generator.py adds image2 and image3 when it runs, and cuts the list to 2 references. A scene with 3 characters loses the third, and a scene with 2 characters leaves no slot for anatomy references in Stage A. Watch for the second character taking the first character's face.
- Output size follows the reference image, not 1280x720. Make sure the keyframe is resized to 16:9 before video, or Wan will crop or stretch.

### 5. Video

- **Lightning is set up correctly** on both Wan experts (high and low noise), 4 steps split 2 and 2. Keep it.
- **Negative prompt is ignored.** cfg is 1.0, so the long Chinese negative list does nothing. Put what you want in the positive prompt.
- **Motion LoRA is attached to the high-noise expert only**, and it was trained on Wan 2.1 text-to-video while you run Wan 2.2 image-to-video. That mismatch is the likeliest reason motion looks weak or the face drifts. See the LoRA section below.
- Scenes are chained from the last frame of the previous video. This keeps continuity but also passes along drift and blur. Re-anchor to the character sheet every few scenes.

### 6. Audio, assembly and operations

- Audio gender uses `any(w in text ...)`, so "she" matches "shelf" and "her" matches "there". Use word boundaries or the tracker's gender.
- Video is 81 frames at 16 fps (5.06 s) but audio length is not measured. Trim or pad audio to the video per scene so the join does not drift.
- The local LLM is not always released before ComfyUI starts, which causes out-of-memory errors. Unload it explicitly and confirm free memory.
- `handler.py` is unfinished and `ffmpeg_utils.py` is empty. Remove or finish them so nobody relies on them.

## Which models to change, and which to keep

| Job | Current model | Decision | Why |
| --- | --- | --- | --- |
| Planner | Qwen3-30B-A3B local, OpenAI fallback | Keep | Works with closed facts. Lower temperature for the facts call. |
| Character sheets | Krea 2 | Keep | Fast (6 steps) and good quality. |
| Scene backgrounds | Qwen-Image-2512 / Juggernaut XL | Keep one | Two backup paths add upkeep. Pick one and delete the other. |
| Keyframes | Qwen-Image-Edit-2511 (bf16, 40 steps) | Keep, speed up | Add a Lightning LoRA or use fp8 after a test run. |
| Video | Wan 2.2 I2V 14B + Lightning | Keep | Correct two-expert setup. Fix the motion LoRA, not the model. |
| Face check | none yet | Add | InsightFace with retry (`identity_check.py`). |
| Voice | OpenAI TTS | Keep | Fix gender and length handling only. |

No model needs replacing now. Change a model only if, after the fixes below, faces still drift. At that point a per-character LoRA is the next step, not a different base model.

## Motion LoRA training review

- **Base mismatch.** Training uses `Wan2.1-T2V-14B`, but inference uses Wan 2.2 image-to-video. A LoRA trained on one base transfers poorly to another. Train against Wan 2.2 I2V (high and low noise) or drop the LoRA until you can.
- **One expert only.** The LoRA is added to the high-noise model. Wan 2.2 splits work between two experts, so motion detail from the low-noise expert is missing.
- **Captions are written by file name.** The caption script guesses the action from folder and file names, so a badly named clip gets a wrong or generic caption. Review a sample of captions by hand.
- **Caption words must match the planner.** Planner acts and caption tags should come from one shared list, or the trigger words will never fire at inference.
- **Resume run changes settings.** The resume script lowers the learning rate (1.5e-4 vs 2e-4) mid-run. That is fine, but record it so results can be compared.
- **Test strength.** The test script uses 0.65. Compare 0.4, 0.65 and 1.0 on the same keyframe and seed before choosing.

A LoRA will help motion style, but it will not by itself fix clothing or identity. That work belongs to the tracker, the per-state references and the face check.

## What to do, in order

1. **This week (small, high impact).**
   - Wire in `identity_check.py` with retry on keyframes.
   - Fix the audio gender match and the hash-based seed.
   - Log every planner fallback and fail loudly on bad output.
   - Remove the keyword fallback from the anatomy check.
   - Unload the local LLM before ComfyUI starts.
2. **Next.**
   - Give each scene its own location and lighting.
   - Test a Qwen-Edit Lightning LoRA for faster keyframes.
   - Match audio length to each video.
   - Resize keyframes to 16:9 and confirm two-character scenes get both references.
3. **Later.**
   - Retrain the motion LoRA on Wan 2.2 I2V for both experts, with captions from the shared act list.
   - Add a per-character LoRA only if faces still drift after the face check.
   - Delete unused files (`handler.py`, empty `ffmpeg_utils.py`, backups).

## Content safety

Two test files (`test_local_planner.py` and `story.txt`) have unclear or underage-looking characters. I will not work with that content, and I recommend deleting those tests. Add a hard rule in the planner: every character must have a stated adult age (18 or over), and the pipeline should stop if one does not.

## Not yet checked

I have not run the full pipeline or the identity check, because the models and InsightFace are not available in my environment. Findings come from reading your code and workflow files, so test each fix on a few scenes before a full run.

## Automating motion-LoRA training

The goal is a flow where you drop raw clips into a folder and get a tested LoRA out, with AI writing the captions and a person only reviewing a small sample. This section covers how training works, what data it needs, how to caption with AI, and how to chain it all.

### How training works, in plain words

1. You show the model many short clips, each with a short text caption.
2. For each clip the model tries to rebuild the motion from noise, guided by the caption. When it gets it wrong, a small add-on (the LoRA) is nudged.
3. After a few thousand steps the add-on has learned "when I see this trigger word and these tags, move like this."
4. At video time you load the add-on next to Wan, and the same words in your motion prompt bring that movement back.

Two consequences follow. First, the caption words at training time must be the same words the planner writes at inference time, or the LoRA never fires. Second, the LoRA learns motion and style, so the clips must show the motion clearly and the captions must describe the motion, not the people.

### What dataset you need

| Item | Recommended | Why |
| --- | --- | --- |
| Clips per action type | 30 to 60 minimum, 100+ for a strong result | Fewer clips make the LoRA copy individual clips instead of learning the motion |
| Clip length | 3 to 5 seconds, one continuous shot | Your config trains on 49 frames; a cut inside a clip teaches jumps |
| Frame rate | Convert all to 16 fps | Matches Wan 2.2 output; mixed rates teach wrong speed |
| Resolution | At least 480p, same aspect ratio (16:9) | You train at 832x480; very small or stretched clips look soft |
| Variety | Different people, rooms, lighting, camera angles per action | If every clip has the same person or room, the LoRA learns the person or room |
| Balance | Similar clip counts per action | A big action drowns out small ones |
| Quality | Sharp, no watermark, no text, no heavy compression | The model copies defects |
| Rights and consent | Only licensed or self-produced material, with every performer verified adult (18+) and consenting | A legal and platform requirement, and a bad dataset cannot be fixed later |

A useful folder layout, one folder per action, with the caption file sitting next to each clip:

```text
dataset/
  action_a/
    clip_0001.mp4
    clip_0001.txt
  action_b/
    clip_0001.mp4
    clip_0001.txt
  _rejected/        # clips the filter threw away, with the reason
  _review/          # clips whose caption the AI was unsure about
```

### Generating captions with AI

Today `generate_all_captions.py` guesses the caption from folder and file names. That fails whenever a file is misnamed, and it cannot see what is in the clip. Replace it with a vision-language model (VLM) that watches the clip. You already run Qwen vision-language models in ComfyUI, so Qwen2.5-VL or Qwen3-VL is the natural choice. A local model also keeps the dataset private.

**How the captioner should work**

1. Take 6 to 8 frames evenly spaced across the clip.
2. Ask the VLM to answer in JSON, choosing only from your fixed vocabulary (the same list the planner uses). Free text is allowed only in one short "motion" sentence.
3. Check the JSON against the vocabulary. Any word outside it is rejected.
4. Compare the AI action with the folder label. If they agree, accept. If they disagree, or the model reports low confidence, move the clip to `_review/`.
5. Build the caption file from the checked fields in a fixed order: trigger word, action tag, camera tag, one motion sentence.

**Prompt and answer format (starting point)**

```text
You label short video clips for training a motion model.
Look at the frames. Describe ONLY the movement and the camera.
Do not describe faces, names, age, or identity.
Answer with JSON only:
{
  "action": one of ACTION_LIST,
  "camera": one of CAMERA_LIST,
  "people_count": 1 | 2 | 3_or_more,
  "motion": one sentence under 20 words about speed and direction of movement,
  "confidence": 0.0 to 1.0
}
If the clip does not clearly show one of the listed actions, set action to "unclear".
```

**Rules that keep captions useful**

- Describe motion, camera and framing. Leave out hair colour, clothing and faces, otherwise the LoRA ties those to the motion.
- Keep one trigger word at the start of every caption, as you do now.
- Keep the vocabulary short (about 10 to 20 action tags and 6 to 10 camera tags). Store it in one file, for example `vocab.json`, and import it in the planner, the captioner and the test script.
- Run the captioner twice with different frame samples and keep the clip only if both runs agree. This cheap check removes most label noise.
- Hand-check 5 to 10 percent of captions, plus everything in `_review/`.

### The automated flow, end to end

1. **Ingest.** Drop clips in an inbox folder. A watcher script picks up new files.
2. **Normalise.** ffmpeg cuts each file at scene changes, trims to 3 to 5 seconds and converts to 16 fps at one resolution.
3. **Filter.** Reject clips that are too short, blurry, watermarked, duplicates (compare frame fingerprints), or have a face too small to use. Write the reason to `_rejected/`.
4. **Caption.** Run the VLM captioner above, validate against `vocab.json`, send unsure clips to `_review/`.
5. **Balance.** Count clips per action. Stop and warn if one action has far fewer than the rest.
6. **Train.** Start AI-Toolkit with a config generated from a template, so settings are never copy-pasted by hand.
7. **Auto-test.** After each saved checkpoint, render the same few keyframes with fixed seeds at strengths 0.4, 0.65 and 1.0.
8. **Score.** Compare each test video with its keyframe: face similarity (reuse `identity_check.py`), amount of motion, and a count of flicker or jump frames.
9. **Register.** Copy the best checkpoint into the ComfyUI loras folder under a versioned name, and record its settings, dataset size and scores in a small log file.

### Training settings to use

| Setting | Starting value | Note |
| --- | --- | --- |
| Base model | Wan 2.2 I2V 14B, high-noise and low-noise trained as two LoRAs | Match what you run at inference; confirm your AI-Toolkit version supports this |
| LoRA rank / alpha | 32 / 16 | As you have now; go to 64 only if motion looks weak |
| Learning rate | 1e-4 to 2e-4 | Lower it if faces start to wobble |
| Steps | 2000 to 3000 | Save every 250 and judge by the test videos, not by loss |
| Frames and size | 49 frames at 832x480 | Fits a single large GPU with quantised weights |
| Captions | Dropout of 5 to 10 percent on the caption | Lets the LoRA also work when the prompt is short |

These are starting points taken from common practice, not results from your data. Treat the first run as a trial and change one setting at a time.

### What this automation will and will not fix

- It removes manual captioning, catches mislabelled clips, and makes runs repeatable and comparable.
- It does not replace clean, varied, rights-cleared data. Caption quality can never be better than the clips behind it.
- Keep a person in the loop for the review folder and for the final choice of checkpoint.

## Implementation and fix plan

The work is split into six phases. Each phase ends with a gate: a check that must pass before you start the next one. Do them in order, because later phases depend on earlier ones (for example, you cannot judge a faster keyframe model without a face check to compare against). Time estimates assume one person working part-time and are rough.

| Phase | Goal | Rough time | Gate to pass |
| --- | --- | --- | --- |
| 0 | Safety net and baseline | 1 day | Baseline videos and scores saved |
| 1 | Quick code fixes | 2 to 3 days | All fixes tested, no new failures |
| 2 | Character consistency | 1 week | Face check passes on most scenes |
| 3 | Planner quality | 1 week | Multi-location stories render correctly |
| 4 | Speed and stability | 1 week | Full story runs unattended without errors |
| 5 | Motion LoRA and captions | 2 to 3 weeks | LoRA beats the no-LoRA baseline |

### Phase 0: Safety net and baseline

Do this first so every later change can be measured and undone.

1. Put the project in git (or copy it) and tag the current state as `baseline`.
2. Delete or quarantine `test_local_planner.py` and `story.txt` as described under Content safety.
3. Create a test set of 5 short stories, each about 6 scenes, that cover: one character, two characters, a location change, a clothing change mid-story, and a night or outdoor scene. Use adult characters with stated ages in every story.
4. Run the current pipeline on all 5 and save the plan, keyframes and videos. This is the baseline you compare against.
5. Create `vocab.json` with the closed lists (acts, events, cameras, genders) that the planner, tracker, captioner and test script will share.

**Gate:** baseline outputs exist for all 5 stories and you can re-run them.

### Phase 1: Quick code fixes

Small, safe changes. Each one is independent, so test and commit them one at a time.

| # | File | Change | Done when |
| --- | --- | --- | --- |
| 1 | `audio_generator.py` | Replace the substring check for gender words with whole-word matching, or read gender from the tracker | "shelf" and "there" no longer set a female voice |
| 2 | `scene_manager.py` | Replace `hash(character_id)` with a stable hash (for example the first 8 hex digits of MD5) | Same seed for the same character after a restart |
| 3 | `continuity.py` | Remove the keyword-substring fallback in `needs_anatomy_pass`; rely on closed acts only | A scene whose text merely contains a word fragment does not trigger the pass |
| 4 | `story_planner.py` | Log every fallback to the job folder and add a strict mode that stops on bad LLM output | A broken plan fails loudly instead of rendering something else |
| 5 | `story_planner.py` | Add an adult-age rule: every character needs an age of 18 or more, else stop | A story with a missing or underage character is rejected |
| 6 | `gpu_memory.py` / `local_llm_provider.py` | Explicitly delete the model, empty the CUDA cache and confirm free memory before ComfyUI starts | No out-of-memory error when planner hands over to ComfyUI |
| 7 | `video_generator.py` | Attach the motion LoRA to both Wan experts, or to none | No half-attached LoRA in the workflow |
| 8 | `handler.py`, `ffmpeg_utils.py` | Finish or delete | No dead files left |

**Gate:** re-run the 5 test stories; no scene gets worse than the baseline, and the error log is empty or explained.

### Phase 2: Character consistency

This phase fixes faces and clothing drifting between scenes.

1. **Wire in the face check.** Register each character's reference sheet in `IdentityChecker`, call `generate_with_retry` around keyframe generation, and keep the best of up to 3 tries. Start with the threshold at 0.38, then adjust using your 5 test stories: raise it if wrong faces pass, lower it if good frames are rejected.
2. **Per-outfit references.** Use `ReferenceBook` so each outfit state of a character has its own reference sheet, and pick the one that matches the tracker's state for that scene.
3. **Two-character scenes.** The keyframe generator fills reference slots image2 and image3 at run time and silently keeps only 2 references in total, while scene\_manager.py allows up to 3 characters. Raise the generator cap, or merge references, and test that neither face swaps.
4. **Resize to 16:9.** Make the keyframe 1280x720 before sending it to Wan, so there is no stretch or crop.
5. **Re-anchor the chain.** Chaining each scene from the previous last frame passes drift forward. Every 3 to 4 scenes, restart from the character reference instead of the last frame.
6. **Fix the seeds.** With stable seeds from Phase 1, keep the same base seed per character and change only the retry offset.

**Gate:** on the 5 test stories, the face check passes on at least 90 percent of keyframes within 3 tries, and clothing state matches the tracker in every scene.

### Phase 3: Planner quality

This phase makes the plan follow the story instead of one fixed room.

1. **Per-scene location.** Let the cast and facts calls return a location for each scene (a closed list the planner builds first, then reuses by id). Remove the hard-coded `location_001` in `build_plan`.
2. **Per-scene lighting and time of day.** Replace the fixed lighting line in `compile_scene` with fields from the facts (for example day, night, indoor, outdoor).
3. **Clean prompts.** Build the keyframe and video prompts only from the closed facts plus the per-character clothing block from the tracker. Do not append the planner's free text.
4. **Validate the plan before spending GPU time.** Check that every character exists, every event is in the closed list, every scene has a location, and every age is 18 or more. Show the problems and stop.
5. **Lower the temperature** for the facts call so the same story gives the same plan.
6. **Keep one background path.** Choose Qwen-Image-2512 or Juggernaut XL for scene backgrounds and remove the other workflow and its code.

**Gate:** a story that moves through three locations renders three different locations, and running the same story twice gives matching plans.

### Phase 4: Speed and stability

1. **Faster keyframes.** Test a 4- or 8-step Lightning LoRA for Qwen-Image-Edit-2511 on 10 scenes. Compare face score and clothing accuracy with the 40-step baseline. Keep it only if the scores stay close.
2. **Match audio to video.** Measure each audio clip. If it is longer than the video, shorten the line or speed it slightly; if shorter, pad with silence. Do this before assembly.
3. **Resume and cleanup.** Make sure a failed job resumes from the last finished scene, and that `job_cleanup.py` never deletes a job that is still running.
4. **Retries and timeouts.** Add a timeout and a limited retry to every ComfyUI call so one stuck scene cannot freeze the story.
5. **Progress and errors.** Write a small status file per job (scene number, stage, last error) that the API can read.

**Gate:** a full 6-scene story runs from paste to final video with no manual step, and a deliberately killed run resumes correctly.

### Phase 5: Motion LoRA and automated captions

Start this only after Phases 1 to 4 pass, because a LoRA cannot be judged on a pipeline that is still drifting.

1. Build the dataset and folder layout from the training section above.
2. Replace `generate_all_captions.py` with the VLM captioner and the review folder.
3. Generate the training config from a template instead of the two copy-pasted shell scripts.
4. Train against Wan 2.2 I2V, one LoRA per expert (high noise and low noise).
5. Run the auto-test at strengths 0.4, 0.65 and 1.0 and score each result.
6. Add the winning LoRA to the pipeline behind a setting, so you can switch it off.

**Gate:** on the 5 test stories, scenes using the LoRA score at least as well on face similarity as scenes without it, and motion is clearly better on the target actions.

### Test and rollback rules for every phase

- Change one thing at a time and re-run the 5 test stories after each change.
- Keep a results table: story, scene, face score, clothing correct (yes/no), time per scene, errors.
- If a change makes any score worse than the baseline, undo it and note why.
- Tag each passed phase in git (`phase-1-done`, and so on) so you can always return to a working state.
- Do not change models and code in the same step, or you cannot tell which one caused a result.

### Definition of done

- Any adult-character story pasted in produces a complete video with no manual step.
- Clothing state matches the tracker in every scene, and the face check passes on at least 90 percent of keyframes.
- Locations and lighting follow the story.
- Every fallback and failure is logged, and a failed run can resume.
- The motion LoRA, if used, was trained on the matching Wan version and tested against a no-LoRA baseline.

## Instructions to give a coding LLM, fix by fix

Use these if you let an LLM make the changes. Paste the **base instruction** first in every session, then one **fix instruction** at a time with the named file attached. Doing one fix per session keeps changes small and easy to check.

### Base instruction (paste first, every time)

```text
You are editing a Python story-to-video pipeline (FastAPI, ComfyUI, Wan 2.2, Qwen).
Rules:
1. Change only what the task names. Do not rename, reformat or refactor anything else.
2. Keep all public function names, arguments and return shapes unchanged unless the task says otherwise.
3. Do not add new dependencies unless the task says so.
4. Never swallow errors with a bare except or a silent default. Log them and either raise or return a clear error.
5. Keep the closed lists (acts, events, genders, cameras) in one place and import them; never copy them.
6. After the change, show a unified diff, then list what you changed, what you did NOT change, and one small test I can run.
7. If something in the task is unclear or conflicts with the code, stop and ask instead of guessing.
```

### Fix instructions

**Fix 1: audio gender (`audio_generator.py`)**

```text
The voice picker uses any(w in full_text for w in ["female","woman","girl","lady","she","her"]), which matches inside other words ("shelf", "there").
Task: use whole-word matching with re.search(r"\b(word)\b", text, re.IGNORECASE). If the character's gender is already known from the continuity tracker or the plan, use that first and use text matching only as a last resort.
Test: "the shelf is there" must NOT pick a female voice; "she smiled" must.
```

**Fix 2: stable seeds (`scene_manager.py`)**

```text
gen_seed uses abs(hash(character_id)), which changes every Python run.
Task: replace hash() with int(hashlib.md5(character_id.encode()).hexdigest()[:8], 16), keeping the same range as now.
Test: print the seed for the same id in two separate runs; both must match.
```

**Fix 3: anatomy-pass fallback (`continuity.py`)**

```text
needs_anatomy_pass has a fallback that scans free text for keyword fragments, which can match inside other words and trigger the pass wrongly.
Task: remove the keyword fallback. Decide using only the closed act from the plan and the tracker state. Keep the function signature.
Test: a scene whose free text contains a fragment but whose act is not in the closed list must return False.
```

**Fix 4: planner fallbacks (`story_planner.py`)**

```text
When the LLM output is invalid the code silently uses defaults.
Task: (a) log every fallback with the scene number, the field and the reason into the job folder in fallbacks.log; (b) add a strict flag (default True) that raises PlanError instead of falling back; (c) include fallback counts in the returned plan under plan["meta"]["fallbacks"].
Test: feed a plan with a missing field; strict mode raises, non-strict logs one line.
```

**Fix 5: adult-age rule (`story_planner.py`)**

```text
Task: in plan validation, require every character to have an integer age of 18 or more. If an age is missing, below 18, or unclear, raise PlanError naming the character and stop before any image or video is generated. Do not guess or default an age.
Test: a cast with age 17, a missing age, and age "young" must each be rejected; age 25 must pass.
```

**Fix 6: free GPU memory (`gpu_memory.py`, `local_llm_provider.py`)**

```text
The local LLM is not always released before ComfyUI starts.
Task: add an unload() to LocalLLMProvider that deletes the model and tokenizer, runs gc.collect() and torch.cuda.empty_cache(), then checks free GPU memory with torch.cuda.mem_get_info(). Call it after planning and before the first ComfyUI call. Log the free memory before and after. If free memory is below a configurable minimum, raise a clear error.
Test: log shows memory freed after unload.
```

**Fix 7: motion LoRA on both experts (`video_generator.py`)**

```text
The motion LoRA is inserted only on the high-noise path (workflow node 7 reads from node 500).
Task: add a second LoraLoaderModelOnly for the low-noise path (node 6 -> node 8), with the same strength, and attach the LoRA to both. If no LoRA name is given, leave the workflow exactly as it is now.
Test: build the workflow with and without a LoRA and print the node graph; with LoRA, both ModelSamplingSD3 nodes must read from a motion-LoRA node.
```

**Fix 8: wire the face check (keyframe generation)**

```text
Host file: app/scene_manager.py, method generate_keyframes (the Stage A / Stage B section). Do NOT put identity logic in keyframe_generator.py; it stays a thin ComfyUI wrapper. identity_check.py is already in app/.

1. Create one IdentityChecker(threshold=0.38) per generate_keyframes call. Import it inside a try block; if insightface is missing, log a warning and skip the check instead of crashing.
2. After the ContinuityTracker is created, call checker.register(char_id, references[char_id]) once for every character in character_map (references[...] is the same reference-sheet path already used for character_refs).
3. Where Stage A and Stage B run, define a local function make_keyframe(seed) that:
   - runs Stage A with seed=seed and writes scene_dir / f"stage_a_try_{seed}.png";
   - if needs_anatomy_pass(...) is true, runs Stage B with seed=seed+1; otherwise copies the Stage A file;
   - writes the final image to scene_dir / f"keyframe_try_{seed}.png" (every attempt gets its own file, so a good earlier attempt is never overwritten);
   - stores the result dict in attempts[seed] and returns the Path of the final image.
4. Call best = checker.generate_with_retry(make_keyframe, expected_ids=scene_character_ids, base_seed=2000003 + order * 7919, max_tries=3). The first attempt must use exactly the current seed, so behaviour without retries is unchanged.
5. Copy best["path"] to keyframe_path with shutil.copy2, copy the winning stage A file to stage_a_keyframe_path, and set keyframe_result = attempts[best["seed"]].
6. Add to the scene metadata: identity_scores = best["report"]["scores"], identity_seed = best["seed"], identity_warning = best["identity_warning"].
7. Delete the losing attempt files after the copy.
8. If every attempt fails, keep the best one, set identity_warning, and continue; do not stop the job.
Test: use a scene whose reference sheet belongs to a different character; the log must show low scores for all tries and identity_warning must be true.
```

**Fix 9: per-scene location and lighting (`story_planner.py`, `compiler.py`)**

```text
build_plan sets location_id = "location_001" for all scenes, and compile_scene uses one fixed lighting line.
Task: add a locations list to the cast call output and a location_id plus a lighting field (closed values such as day, night, indoor, outdoor) to the per-scene facts. Pass them through to the compiler and use them instead of the fixed values. Keep a default only when the field is absent, and log that default.
Test: a story with a street scene then a bedroom scene produces two different location ids and different lighting text.
```

**Fix 10: audio length (`audio_generator.py`, `video_utils.py`)**

```text
Task: after making each scene's audio, measure its length with ffprobe. If longer than the video, speed it up by at most 10 percent; if still longer, trim with a short fade-out and log a warning. If shorter, pad with silence to the video length. Do this before video_with_audio.mp4 is built.
Test: a 7-second audio and a 5-second video give a 5-second final scene.
```

### How to review what the LLM gives you

1. Read the diff first. If it touches files or functions the task did not name, reject it and repeat the task with "change only the named function."
2. Run the small test it gave you, then run one of your 5 test stories.
3. Check the new logs: you should see fallbacks, scores or memory numbers, not silence.
4. Commit each fix on its own, with the fix number in the message.
5. If two fixes interact (for example Fix 8 and Fix 2), run the face-check test again after both are in.

### Things an LLM often gets wrong here

- It may "improve" prompts or rename keys, which breaks the planner-to-compiler contract. Rule 2 above guards against this.
- It may add a try/except that hides an error. Rule 4 guards against this.
- It may invent function names in `identity_check.py` or ComfyUI nodes. Always attach the real file and check the diff against it.
- It may treat every vocabulary word as free text. Insist that lists come from `vocab.json`.

## Extra suggestions not covered above

These are the remaining ideas from the review. They are mostly workflow-level checks and optional upgrades, so they sit after the main phases.

### Workflow file checks (ComfyUI JSON)

| Workflow | What I saw | What to do |
| --- | --- | --- |
| `krea2_t2i_api.json` | Contains a hard-coded example prompt (detective in an observatory) and a negative prompt at cfg 1.0 | Confirm the code always overwrites the prompt node; a failed overwrite would silently render the example. Drop the negative prompt or raise cfg if you want it used |
| `qwen_image_edit_2511_api.json` | `Template has image1 only, and the code adds image2/image3 (max 2 references`); the negative-side text node holds a default "maintain identity" sentence; the starting latent comes from the reference image at denoise 1.0 | Confirm what text your code writes into the negative node. Test a real negative or an empty one. Check output size, since it follows the reference image |
| `wan2.2_i2v_api.json` | Lightning on both experts, 4 steps split 2 and 2, cfg 1.0, fixed 1280x720, 81 frames at 16 fps | Keep. Remember the negative prompt has no effect at cfg 1.0 |
| `wan2.2_t2v_14b_api.json` | Has a Lightning on/off switch that is set to off (20 steps, cfg 3.5) | Confirm whether the code uses this file at all. If not, delete it |
| `juggernaut_xl_t2i_api.json` | Empty prompts, 35 steps, SDXL checkpoint | Confirm it is still used. If Qwen-Image-2512 replaced it, delete it |
| `qwen_image_2512_api.json` | 50 steps, cfg 4, placeholder prompt | Keep one background workflow only (see Phase 3) |

Rule for all workflows: after the code patches a workflow, check that no placeholder text (`PLACEHOLDER_PROMPT`, `MOTION_PROMPT`, example prompts) is left. Fail the job if one is.

### Optional upgrades

1. **Clothing check after each keyframe.** Ask a vision-language model one closed question per character, for example "is the top worn: yes or no", and compare with the tracker. Retry when they differ. This catches the exact failure that started this project and does not depend on the planner.
2. **Character LoRA.** If faces still drift after the face check and per-outfit references, train a small LoRA per main character. It is heavier work, so keep it for recurring characters.
3. **One shared `vocab.json`.** Use it in the planner, tracker, captioner and test script so a word means the same thing everywhere.
4. **Plan preview.** Show the compiled plan (scenes, locations, clothing states) before generation starts, so you can fix a bad plan before spending GPU time.
5. **Per-scene cost log.** Record seconds per stage per scene. This shows where time goes and whether a speed-up really helped.

### Expectations about the motion LoRA

- It can improve how a motion looks and moves. It cannot, by itself, stop clothing flips or fix a wrong face.
- A result can only be trusted after a fixed-seed comparison with and without the LoRA on your test stories.
- Treat any claim of a guaranteed fix, from me or anyone else, with caution until you have measured it.

## Master checklist

This list is the single place to track everything. Tick items as you finish them. If you or I add a new suggestion later, it goes here first.

### Do first (Phases 0 and 1)

- [ ] Tag current code as `baseline` in git
- [ ] Remove the two unsafe test files
- [ ] Create the 5 adult-character test stories and save baseline outputs
- [ ] Create `vocab.json`
- [ ] Fix 1: audio gender whole-word match
- [ ] Fix 2: stable seed hash
- [ ] Fix 3: remove anatomy keyword fallback
- [ ] Fix 4: log planner fallbacks, add strict mode
- [ ] Fix 5: adult-age rule (18 or over)
- [ ] Fix 6: unload local LLM before ComfyUI
- [ ] Fix 7: motion LoRA on both Wan experts, or none
- [ ] Delete or finish `handler.py` and `ffmpeg_utils.py`

### Consistency (Phase 2)

- [ ] Fix 8: wire in the face check with retry
- [ ] Per-outfit reference sheets in use
- [ ] Two-character scenes keep both faces
- [ ] Keyframes resized to 16:9
- [ ] Re-anchor to the reference every 3 to 4 scenes

### Planner (Phase 3)

- [ ] Fix 9: per-scene location and lighting
- [ ] Prompts built only from closed facts
- [ ] Plan validation before any GPU work
- [ ] Lower temperature for the facts call
- [ ] Keep one background workflow, delete the other

### Speed and stability (Phase 4)

- [ ] Test Qwen Edit Lightning LoRA against the 40-step baseline
- [ ] Fix 10: audio length matched to video
- [ ] Resume after failure works
- [ ] Timeouts and limited retries on every ComfyUI call
- [ ] Per-job status file
- [ ] Placeholder-text check on every patched workflow

### Motion LoRA (Phase 5)

- [ ] Dataset built, rights and adult-age verified
- [ ] VLM captioner replaces `generate_all_captions.py`
- [ ] Training config generated from a template
- [ ] Trained on Wan 2.2 I2V, both experts
- [ ] Auto-test at strengths 0.4, 0.65 and 1.0
- [ ] LoRA beats the no-LoRA baseline

### Optional

- [ ] Clothing check with a vision-language model
- [ ] Character LoRA for recurring characters
- [ ] Plan preview before generation
- [ ] Per-scene cost log

## Using Antigravity: which model, and the master prompt

The model list you showed has four options. These picks come from general strengths of each model family, not from a test on your code, so run Fix 1 with two models and compare before committing to one.

| Model | Use it for | Avoid it for |
| --- | --- | --- |
| Claude Opus 4.6 (Thinking) | Changes that touch several files or need careful reasoning: Fix 4, 5, 8, 9 (planner, tracker, face check, per-scene location) | Tiny edits, where it costs more quota than needed |
| Claude Sonnet 4.6 (Thinking) | Small, single-file fixes: Fix 1, 2, 3, 6, 7, 10 and cleanup | Large cross-file redesigns |
| Gemini 3.8 Flash | Fast, mechanical work: deleting dead files, renaming, writing test stories, reading a whole folder | Anything where a subtle logic mistake would be costly |
| GPT-OSS 120B | Not recommended here | Planner, tracker and ComfyUI wiring |

The "Notice" label next to some models is likely a usage or quota warning. Click it to see what it says before a long session.

### How to run it

1. Save this document in the repo, for example as `docs/PIPELINE_REVIEW.md`, and commit it.
2. Open a new Antigravity session for each fix. Do not chain several fixes in one long chat.
3. Paste the master prompt below, then name the fix number.
4. Read the diff, run the test it gives you, commit, then start the next session.

### Master prompt (paste at the start of each session)

```text
You are fixing a Python story-to-video pipeline (FastAPI, ComfyUI, Wan 2.2, Qwen).
Read docs/PIPELINE_REVIEW.md first. It is the single source of truth.

Today's task: implement ONLY "Fix <NUMBER>" from the section
"Instructions to give a coding LLM, fix by fix".

Rules:
1. Read the named files fully before editing. Do not guess function names; open them.
2. Change only what the fix names. No refactors, renames or formatting changes elsewhere.
3. Keep public function names, arguments and return shapes unchanged.
4. No new dependencies unless the fix says so.
5. Never hide errors with a bare except or silent default. Log and raise.
6. Closed lists (acts, events, genders, cameras) come from vocab.json. Do not copy them.
7. Before editing, write a 5-line plan: files to touch, what changes, what stays.
8. After editing, show the diff, then list: what changed, what you did NOT change, and one test command I can run.
9. If the doc and the code disagree, stop and ask me.
10. Tick the matching item in the Master checklist only after I confirm the test passed.
```

### Safe-edit habits in Antigravity

- Work on a git branch per fix (`fix-03-anatomy-fallback`), so a bad change is one command to undo.
- Ask the model to run the test itself if the tool allows terminal access, but still read the output yourself.
- If the model starts editing files you did not name, stop the run and restate rule 2.
- If a model keeps failing the same fix twice, switch to Opus 4.6 for that fix rather than retrying the same model.
