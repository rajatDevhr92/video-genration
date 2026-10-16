import json
import logging
from typing import Dict, Any, List, Set, Optional

logger = logging.getLogger(__name__)

ACTS = ["none", "kiss", "touch", "oral", "vaginal", "anal", "other_contact"]
CONTACT = ["none", "approaching", "engaged_still", "engaged_moving", "separating"]
PACE = ["very_slow", "slow", "medium", "fast"]
WARDROBE = ["remove_top", "remove_bottom", "remove_all", "dress_all"]

ACT_ALIASES = {
    "kissing": "kiss",
    "kisses": "kiss",
    "caress": "touch",
    "touching": "touch",
    "caresses": "touch",
    "blowjob": "oral",
    "cunnilingus": "oral",
    "eating out": "oral",
    "oral sex": "oral",
    "sex": "vaginal",
    "intercourse": "vaginal",
    "penetration": "vaginal",
    "thrusting": "vaginal",
    "fucking": "vaginal",
    "grinding": "other_contact",
    "dry humping": "other_contact",
}

CONTACT_ALIASES = {
    "contact": "engaged_moving",
    "engaged": "engaged_moving",
    "touching": "engaged_moving",
    "still": "engaged_still",
    "holding": "engaged_still",
    "apart": "none",
    "separate": "separating",
    "close": "approaching",
}

FACTS_PROMPT = """Read ONE scene from a story and answer with JSON only.
Use only facts stated in the scene text. If a fact is not stated, use the value of the previous scene.

Previous scene facts: {prev}
Characters (id: name, gender): {cast}
Scene text: {text}

Return JSON with keys:
act: one of {acts}
contact_state: one of {contact}
pace: one of {pace}
pose_text: <=8 words, body posture of each character as stated
same_pose_as_previous: true or false (true if the text says the position stays or nothing indicates a change)
character_ids: list of character ids present in this scene
wardrobe_events: list of {{"character_id": "...", "event": "..."}} where event is one of {wardrobe}; [] if none
hands: <=15 words, hand placement if stated, else ""
"""


def clean(
    f: Dict[str, Any],
    cast_ids: Set[str],
    cast: List[Dict[str, Any]],
    prev: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validates and normalizes extracted scene facts against allowed enums.
    """
    errs = []

    # 1. Normalize Act
    act = str(f.get("act", "")).lower().strip()
    if act in ACT_ALIASES:
        act = ACT_ALIASES[act]
    if act not in ACTS:
        errs.append(f"invalid act '{act}' (must be one of {ACTS})")
    f["act"] = act

    # 2. Normalize Contact State
    contact = str(f.get("contact_state", "")).lower().strip()
    if contact in CONTACT_ALIASES:
        contact = CONTACT_ALIASES[contact]
    if contact not in CONTACT:
        errs.append(f"invalid contact_state '{contact}' (must be one of {CONTACT})")
    f["contact_state"] = contact

    # 3. Normalize Pace
    pace = str(f.get("pace", "")).lower().strip()
    if pace not in PACE:
        pace = "medium"
    f["pace"] = pace

    # 4. Normalize same_pose_as_previous
    sp = f.get("same_pose_as_previous")
    if isinstance(sp, str):
        sp = sp.lower() in ("true", "1", "yes")
    elif isinstance(sp, (int, float)):
        sp = bool(sp)
    elif not isinstance(sp, bool):
        sp = False if prev is None else True
    f["same_pose_as_previous"] = sp

    # 5. Normalize character_ids
    name_to_id = {c["name"].lower(): c["id"] for c in cast if "name" in c and "id" in c}
    raw_ids = f.get("character_ids", [])
    if not isinstance(raw_ids, list):
        raw_ids = [raw_ids]

    valid_cids = []
    for cid in raw_ids:
        cid_str = str(cid).strip()
        if cid_str in cast_ids:
            valid_cids.append(cid_str)
        elif cid_str.lower() in name_to_id:
            valid_cids.append(name_to_id[cid_str.lower()])

    if not valid_cids:
        valid_cids = prev.get("character_ids", list(cast_ids)) if prev else list(cast_ids)
    f["character_ids"] = valid_cids

    # 6. Normalize wardrobe_events
    ev = f.get("wardrobe_events", [])
    if not isinstance(ev, list):
        ev = []
    clean_ev = []
    for e in ev:
        if isinstance(e, dict):
            c_id = str(e.get("character_id", "")).strip()
            if c_id.lower() in name_to_id:
                c_id = name_to_id[c_id.lower()]
            event_type = str(e.get("event", "")).lower().strip()
            if c_id in cast_ids and event_type in WARDROBE:
                clean_ev.append({"character_id": c_id, "event": event_type})
    f["wardrobe_events"] = clean_ev

    # 7. Clamp text lengths
    f["pose_text"] = str(f.get("pose_text", ""))[:80].strip()
    f["hands"] = str(f.get("hands", ""))[:120].strip()

    if errs:
        raise ValueError("; ".join(errs))

    return f


def extract_facts(
    provider: Any,
    scene: Dict[str, Any],
    cast: List[Dict[str, Any]],
    prev: Optional[Dict[str, Any]] = None,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Queries the LLM for structured closed-enum facts for a single scene with local retries
    and deterministic fallback inheritance.
    """
    cast_ids = {c["id"] for c in cast}
    prompt = FACTS_PROMPT.format(
        prev=json.dumps(prev, ensure_ascii=False) if prev else "none (first scene)",
        text=scene.get("text", ""),
        cast="; ".join(f"{c.get('id')}: {c.get('name')}, {c.get('gender')}" for c in cast),
        acts=ACTS,
        contact=CONTACT,
        pace=PACE,
        wardrobe=WARDROBE,
    )

    last_error = None
    for attempt in range(retries + 1):
        try:
            raw = provider.generate_json(prompt, temperature=0.2)
            if not isinstance(raw, dict):
                raise ValueError("Output must be a JSON object.")
            return clean(raw, cast_ids, cast, prev)
        except Exception as e:
            last_error = e
            logger.warning(
                f"[FACTS] Scene {scene.get('order')} attempt {attempt + 1} validation failed: {e}"
            )
            prompt += f"\nYour last answer was invalid ({e}). Fix only that and return JSON."

    # Safe fallback: inherit from previous scene facts or provide initial default
    logger.warning(
        f"[FACTS] Scene {scene.get('order')} falling back to inherited/default state after errors: {last_error}"
    )
    if prev:
        return {
            "act": prev.get("act", "none"),
            "contact_state": prev.get("contact_state", "none"),
            "pace": prev.get("pace", "medium"),
            "pose_text": prev.get("pose_text", "natural pose"),
            "same_pose_as_previous": True,
            "character_ids": prev.get("character_ids", list(cast_ids)),
            "wardrobe_events": [],
            "hands": prev.get("hands", ""),
        }
    return {
        "act": "none",
        "contact_state": "none",
        "pace": "medium",
        "pose_text": "natural pose",
        "same_pose_as_previous": False,
        "character_ids": list(cast_ids),
        "wardrobe_events": [],
        "hands": "",
    }

