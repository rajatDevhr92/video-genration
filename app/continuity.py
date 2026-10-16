"""
continuity.py  -  deterministic per-character state tracking.

The LLM planner only supplies STRUCTURED facts per scene:
    wardrobe_events: [{"character_id": "...", "event": "remove_top|remove_bottom|remove_all|dress_all"}]
    act:             one of ACTS
Everything else (clothing text, anatomy locks, negative prompts) is compiled
here from state, never guessed from keywords.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ACTS = {"none", "kiss", "touch", "oral", "vaginal", "anal", "other_contact"}
EVENTS = {"remove_top", "remove_bottom", "remove_all", "dress_all"}

_FEMALE = re.compile(r"\b(female|woman|women|girl|lady|she|her)\b", re.I)
_MALE = re.compile(r"\b(male|man|men|guy|boy|he|him|his)\b", re.I)


def normalize_gender(*texts: str) -> str:
    """Single source of truth. Word-boundary match, female checked first.
    Call ONCE per character when the plan is loaded and store the result."""
    blob = " ".join(t for t in texts if t)
    if _FEMALE.search(blob):
        return "female"
    if _MALE.search(blob):
        return "male"
    raise ValueError(f"Cannot determine gender from: {blob!r}. "
                     "Add an explicit 'gender' field to the character.")


@dataclass
class CharState:
    character_id: str
    name: str
    gender: str
    outfit: str                 # canonical outfit text from character sheet
    top_on: bool = True
    bottom_on: bool = True

    @property
    def nude(self) -> bool:
        return not self.top_on and not self.bottom_on

    def clothing_text(self) -> str:
        if self.nude:
            return "fully nude, bare skin, no clothing or underwear"
        if self.top_on and self.bottom_on:
            return f"wearing: {self.outfit}"
        if self.top_on:
            return f"wearing the top of: {self.outfit}; lower body bare"
        return f"wearing the bottom of: {self.outfit}; upper body bare"

    def apply(self, event: str):
        if event == "remove_top":
            self.top_on = False
        elif event == "remove_bottom":
            self.bottom_on = False
        elif event == "remove_all":
            self.top_on = self.bottom_on = False
        elif event == "dress_all":
            self.top_on = self.bottom_on = True


class ContinuityTracker:
    def __init__(self, characters: List[dict]):
        self.state: Dict[str, CharState] = {}
        for c in characters:
            gender = normalize_gender(
                c.get("gender", ""), c.get("visual_anchor", ""), c.get("appearance", ""), c.get("name", ""))
            outfit = c.get("clothing", "casual outfit")
            is_initially_nude = any(w in outfit.lower() for w in ["nude", "unclothed", "naked", "bare skin"])
            self.state[c["id"]] = CharState(
                character_id=c["id"], name=c.get("name", c["id"]),
                gender=gender, outfit=outfit,
                top_on=not is_initially_nude,
                bottom_on=not is_initially_nude,
            )

    # ---- called once per scene, in order -------------------------------
    def advance(self, scene: dict) -> Dict[str, CharState]:
        raw_events = scene.get("wardrobe_events")
        if raw_events is None or (isinstance(raw_events, list) and not raw_events):
            events = self._fallback_events(scene)
        else:
            events = self._normalize_events(raw_events, scene)

        for ev in events:
            if isinstance(ev, dict):
                cs = self.state.get(ev.get("character_id"))
                event_name = ev.get("event")
                if cs and event_name in EVENTS:
                    cs.apply(event_name)

        # Preserve normalized structured events in the scene
        scene["wardrobe_events"] = events
        scene_cids = scene.get("character_ids") or list(self.state.keys())
        scene["clothing_state_by_char"] = {
            cid: self.state[cid].clothing_text()
            for cid in scene_cids if cid in self.state}
        return {cid: self.state[cid] for cid in scene_cids
                if cid in self.state}

    def _normalize_events(self, raw_events, scene: dict) -> List[Dict[str, str]]:
        if isinstance(raw_events, (dict, str)):
            raw_list = [raw_events]
        elif isinstance(raw_events, (list, tuple)):
            raw_list = list(raw_events)
        else:
            return []

        text = f"{scene.get('description', '')} {scene.get('action', '')} {scene.get('primary_motion', '')}"
        scene_cids = [cid for cid in scene.get("character_ids", []) if cid in self.state] or list(self.state.keys())
        normalized = []

        for item in raw_list:
            cid = None
            ev_name = None

            if isinstance(item, dict):
                cid = item.get("character_id") or item.get("character") or item.get("id")
                ev_name = item.get("event") or item.get("action") or item.get("type")
            elif isinstance(item, str):
                s = item.strip()
                if ":" in s:
                    parts = s.split(":", 1)
                    cid, ev_name = parts[0].strip(), parts[1].strip()
                elif " - " in s:
                    parts = s.split(" - ", 1)
                    cid, ev_name = parts[0].strip(), parts[1].strip()
                else:
                    ev_name = s

            # Resolve canonical event name
            if ev_name:
                cleaned_ev = str(ev_name).lower().strip().replace("-", "_").replace(" ", "_")
                if cleaned_ev in EVENTS:
                    ev_name = cleaned_ev
                elif any(k in cleaned_ev for k in ("remove_all", "undress", "strip", "nude", "naked")):
                    ev_name = "remove_all"
                elif any(k in cleaned_ev for k in ("dress_all", "clothe", "get_dressed")):
                    ev_name = "dress_all"
                elif any(k in cleaned_ev for k in ("remove_top", "shirt", "blouse", "bra", "jacket", "top")):
                    ev_name = "remove_top"
                elif any(k in cleaned_ev for k in ("remove_bottom", "pants", "trousers", "skirt", "shorts", "thong", "panties", "underwear", "boxers", "briefs", "bottom")):
                    ev_name = "remove_bottom"
                else:
                    ev_name = None

            if not ev_name:
                continue

            # Match character ID or Name
            target_cid = None
            if cid:
                c_str = str(cid).strip().lower()
                for scid, cs in self.state.items():
                    if scid.lower() == c_str or cs.name.lower() == c_str:
                        target_cid = scid
                        break

            if not target_cid:
                if len(scene_cids) == 1:
                    target_cid = scene_cids[0]
                else:
                    # 1. Possessive check in scene text: e.g. "Sunny's blouse", "Jhone's shirt"
                    for scid in scene_cids:
                        cname = self.state[scid].name.lower()
                        if re.search(rf"\b{re.escape(cname)}'s\b", text, re.I):
                            target_cid = scid
                            break

                    # 2. Gendered pronoun check in scene text
                    if not target_cid:
                        females = [c for c in scene_cids if self.state[c].gender == "female"]
                        males = [c for c in scene_cids if self.state[c].gender == "male"]
                        has_fem = bool(re.search(r"\b(her|she)\s+(?:[a-z]+\s+)*(?:shirt|blouse|top|jacket|dress|bra|pants|trousers|skirt|shorts|thong|panties|underwear|boxers|briefs|shoulders|arms|hips|thighs|legs|body)\b", text, re.I))
                        has_masc = bool(re.search(r"\b(his|he|him)\s+(?:[a-z]+\s+)*(?:shirt|blouse|top|jacket|dress|bra|pants|trousers|skirt|shorts|thong|panties|underwear|boxers|briefs|shoulders|arms|hips|thighs|legs|body)\b", text, re.I))
                        if has_fem and not has_masc and females:
                            target_cid = females[0]
                        elif has_masc and not has_fem and males:
                            target_cid = males[0]

                    # 3. Check current state (which character still has this clothing item on)
                    if not target_cid:
                        if ev_name == "remove_top":
                            wearing = [c for c in scene_cids if self.state[c].top_on]
                            if len(wearing) == 1:
                                target_cid = wearing[0]
                        elif ev_name == "remove_bottom":
                            wearing = [c for c in scene_cids if self.state[c].bottom_on]
                            if len(wearing) == 1:
                                target_cid = wearing[0]
                        elif ev_name == "remove_all":
                            wearing = [c for c in scene_cids if not self.state[c].nude]
                            if len(wearing) == 1:
                                target_cid = wearing[0]

                    if not target_cid:
                        target_cid = scene_cids[0]

            normalized.append({"character_id": target_cid, "event": ev_name})

        return normalized

    # ---- last-resort fallback: sentence-level, word-boundary only ------
    _VERB = re.compile(
        r"\b(removes?|takes? off|slips? off|unbuttons?|unzips?|pulls? off|"
        r"strips?|undresses|pulls? (?:the \w+ )?down|pulls? down|slides? down|slips? down)\b", re.I)
    _DRESS = re.compile(r"\b(gets dressed|puts on (?:his|her|their) clothes)\b", re.I)

    def _fallback_events(self, scene):
        out = []
        text = f"{scene.get('description', '')}. {scene.get('action', '')}"
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            subj = [cid for cid in scene.get("character_ids", [])
                    if re.search(rf"\b{re.escape(self.state[cid].name)}\b", sentence)]
            if not subj:  # fall back to pronoun by gender
                for cid in scene.get("character_ids", []):
                    g = self.state[cid].gender
                    pr = r"\b(she|her)\b" if g == "female" else r"\b(he|his)\b"
                    if re.search(pr, sentence, re.I):
                        subj.append(cid)
            if self._DRESS.search(sentence):
                out += [{"character_id": c, "event": "dress_all"} for c in subj]
            elif self._VERB.search(sentence):
                s = sentence.lower()
                ev = ("remove_top" if re.search(r"\b(shirt|blouse|top|jacket|dress|bra)\b", s)
                      else "remove_bottom" if re.search(r"\b(pants|trousers|skirt|shorts|thong|panties|underwear|boxers|briefs)\b", s)
                      else "remove_all")
                out += [{"character_id": c, "event": ev} for c in subj]
        return out


# ---- prompt compilation (replaces the keyword-driven blocks) -------------
def anatomy_lock(states: Dict[str, CharState]) -> str:
    lines = []
    for cs in states.values():
        if cs.gender == "female":
            lines.append(f"{cs.name}: female body only. No male genitals on {cs.name}.")
        else:
            lines.append(f"{cs.name}: male body only. No female genitals or women's clothing on {cs.name}.")
    return "\n".join(lines)


def per_character_clothing_block(states: Dict[str, CharState]) -> str:
    return "\n".join(f"- {cs.name} ({cs.gender}): {cs.clothing_text()}"
                     for cs in states.values())


def dynamic_negative(states: Dict[str, CharState]) -> str:
    neg = ["bad anatomy", "extra limbs", "fused bodies", "text", "watermark"]
    for cs in states.values():
        if cs.gender == "female":
            neg.append(f"male genitals on {cs.name}")
        else:
            neg.append(f"female genitals or bra or women's clothing on {cs.name}")
        if cs.nude:
            neg.append(f"clothing or underwear on {cs.name}")
        elif cs.top_on and cs.bottom_on:
            neg.append(f"nudity on {cs.name}")
    return ", ".join(neg)


class ReferenceBook:
    """
    One reference sheet per (character, wardrobe bucket).
    Buckets: "dressed" (top and bottom on, or only partly undressed)
             "undressed" (both off)
    Falls back to the base sheet when a bucket was not generated.
    """

    def __init__(self, base_refs: Dict[str, str]):
        self.base = dict(base_refs)               # character_id -> path
        self.by_state: Dict[tuple, str] = {}

    @staticmethod
    def bucket(cs: "CharState") -> str:
        return "undressed" if cs.nude else "dressed"

    def add(self, character_id: str, bucket: str, path: str):
        self.by_state[(character_id, bucket)] = path

    def get(self, cs: "CharState") -> str:
        return self.by_state.get((cs.character_id, self.bucket(cs)),
                                 self.base[cs.character_id])

    def refs_for_scene(self, states: Dict[str, "CharState"]) -> List[str]:
        return [self.get(cs) for cs in states.values()]


def build_reference_book(job_id, characters, base_refs, generator, tracker_states,
                         undressed_ids):
    """
    Generates an extra 'undressed' sheet only for characters who actually
    undress somewhere in the story (undressed_ids), using your existing
    CharacterReferenceGenerator. NOTE: check that generator.generate() writes to
    a path that includes something unique per call; if it always writes
    character_<id>.png it will overwrite the base sheet - copy the result
    elsewhere right after the call (done below).
    """
    import shutil
    from pathlib import Path
    book = ReferenceBook(base_refs)
    for i, c in enumerate(characters, start=1):
        if c["id"] not in undressed_ids:
            continue
        variant = dict(c)
        variant["clothing"] = "Fully nude, completely unclothed, bare skin"
        base_copy = Path(base_refs[c["id"]])
        keep = base_copy.with_name(base_copy.stem + "_dressed" + base_copy.suffix)
        shutil.copy2(base_copy, keep)                      # protect the original
        res = generator.generate(job_id=job_id, character=variant,
                                 seed=1000003 + i * 7919 + 17)
        undressed_path = base_copy.with_name(base_copy.stem + "_undressed" + base_copy.suffix)
        shutil.copy2(res["path"], undressed_path)
        shutil.copy2(keep, base_copy)                      # restore original if overwritten
        book.add(c["id"], "undressed", str(undressed_path))
        book.add(c["id"], "dressed", str(keep))
    return book


def characters_who_undress(scenes: List[dict], tracker_factory) -> set:
    """Dry-run the tracker over the whole plan to find who ever becomes undressed."""
    tracker = tracker_factory()
    out = set()
    for s in scenes:
        for cid, cs in tracker.advance(dict(s)).items():
            if cs.nude:
                out.add(cid)
    return out


def needs_anatomy_pass(act: str, states: Dict[str, CharState], scene: Optional[dict] = None) -> bool:
    """Stage B only when an act involving exposed anatomy is happening
    AND the relevant characters are actually unclothed - not keyword guessing."""
    act_str = str(act).lower() if act else ""
    if act_str not in {"oral", "vaginal", "anal"}:
        if scene:
            pen_text = f"{scene.get('penetration_state', '')} {scene.get('description', '')} {scene.get('action', '')}".lower()
            if any(w in pen_text for w in ["penetrat", "thrust", "cock", "dick", "pussy", "vagina", "anal", "oral", "doggystyle", "cowgirl"]):
                return any(not cs.bottom_on for cs in states.values())
        return False
    return any(not cs.bottom_on for cs in states.values())