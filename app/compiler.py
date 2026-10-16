import re
from typing import Dict, Any, List, Optional

__all__ = ["compile_camera", "compile_scene", "enforce_wardrobe"]

SIZES = [
    ("extreme close-up", r"extreme\s+close"),
    ("close-up", r"close|tight"),
    ("medium", r"medium|mid"),
    ("wide", r"wide|full"),
]

ANGLES = [
    ("rear", r"\b(rear|behind|back)\b", "RIGHT PANEL Rear View"),
    ("side", r"\b(side|profile|3/4)\b", "CENTER PANEL 3/4 Side View"),
    ("front", r"\bfront", "LEFT PANEL Front View"),
    ("overhead", r"\b(overhead|top[- ]down)\b", "LEFT PANEL Front View"),
]

ACT_MOTION = {
    "none": "natural idle movement",
    "kiss": "slow kissing movement",
    "touch": "slow caressing movement",
    "oral": "steady rhythmic head and mouth movement",
    "vaginal": "steady rhythmic hip movement",
    "anal": "steady rhythmic hip movement",
    "other_contact": "gentle shifting movement",
}

PACE_PHRASE = {
    "very_slow": "very slow",
    "slow": "slow",
    "medium": "moderate",
    "fast": "fast",
}

CONTACT_PHRASE = {
    "none": "no body contact",
    "approaching": "bodies moving into contact",
    "engaged_still": "bodies in firm contact, held still",
    "engaged_moving": "bodies in continuous contact while moving",
    "separating": "bodies moving apart",
}

DEFAULT_HANDS = "hands resting naturally on body or mattress"

EXPOSED_ACTS = {"oral", "vaginal", "anal"}


def compile_camera(hint: str, fallback_angle: str = "side") -> Dict[str, str]:
    """
    Deterministically compiles camera shot size, angle, and 3D turnaround panel mapping
    directly from the scene shot hint or text.
    """
    h = (hint or "").lower()
    size = next((n for n, rx in SIZES if re.search(rx, h)), "medium")
    ang = next(((n, p) for n, rx, p in ANGLES if re.search(rx, h)), None)
    if ang is None:
        ang = next(
            ((n, p) for n, _, p in ANGLES if n == fallback_angle),
            ("side", "CENTER PANEL 3/4 Side View"),
        )
    return {
        "size": size,
        "angle": ang[0],
        "turnaround_panel": ang[1],
        "text": f"{size.capitalize()} shot, {ang[0]} angle (matches {ang[1]} of turnaround sheet)",
    }


def compile_scene(
    scene: Dict[str, Any],
    facts: Dict[str, Any],
    prev_compiled: Optional[Dict[str, Any]],
    cast_by_id: Dict[str, Dict[str, Any]],
    location_id: str = "location_001",
) -> Dict[str, Any]:
    """
    Deterministically compiles a full production scene dictionary from raw scene text
    and closed-enum facts, ensuring schema completeness and continuity stability.
    """
    order = int(scene.get("order", 1))
    prev_pose = prev_compiled.get("pose_lock") if prev_compiled else None
    same_pose = bool(prev_pose) and bool(facts.get("same_pose_as_previous", False))
    pose_lock = prev_pose if same_pose else (facts.get("pose_text") or prev_pose or "natural pose")

    active_char_ids = facts.get("character_ids", list(cast_by_id.keys()))
    names_list = [cast_by_id[cid]["name"] for cid in active_char_ids if cid in cast_by_id]
    names_str = " and ".join(names_list) if names_list else "Characters"

    hands = facts.get("hands") or (
        prev_compiled.get("hands") if same_pose and prev_compiled else DEFAULT_HANDS
    )

    cam_hint = scene.get("shot_hint") or facts.get("camera_hint") or ""
    cam = compile_camera(cam_hint)

    act = facts.get("act", "none")
    contact_state = facts.get("contact_state", "none")
    pace = facts.get("pace", "medium")

    motion_on = contact_state in ("engaged_moving", "approaching", "separating") or act in (
        "kiss",
        "touch",
        "oral",
        "vaginal",
        "anal",
    )
    pace_desc = PACE_PHRASE.get(pace, "moderate")
    act_desc = ACT_MOTION.get(act, "natural movement")

    motion = (
        f"{names_str}: {pace_desc} {act_desc}"
        if motion_on
        else f"{names_str}: minimal movement, holding position"
    )

    contact_desc = CONTACT_PHRASE.get(contact_state, "no body contact").capitalize()
    video_prompt = (
        f"{motion}. Exact contact: {contact_desc}. Hands: {hands}. "
        f"Bodies remain locked in {pose_lock}. Camera: {cam['text']}."
    )

    compiled: Dict[str, Any] = {
        "id": f"scene_{order:03d}",
        "order": order,
        "duration": float(scene.get("duration", 5.0)),
        "location_id": location_id,
        "character_ids": active_char_ids,
        "description": scene.get("text", ""),
        "action": facts.get("pose_text") or scene.get("text", ""),
        "camera": cam["text"],
        "camera_size": cam["size"],
        "camera_angle": cam["angle"],
        "turnaround_panel": cam["turnaround_panel"],
        "lighting": "Soft warm indoor cinematic lighting, shallow depth of field",
        "video_prompt": video_prompt,
        "pose_lock": pose_lock,
        "pose_changed": not same_pose,
        "hands": hands,
        "hand_positions": hands,
        "contact_zone": contact_state,
        "contact_state": contact_state,
        "motion_intensity": pace.replace("_", " "),
        "primary_motion": motion,
        "act": act,
        "wardrobe_events": list(facts.get("wardrobe_events", [])),
        "expression": facts.get("expression", "natural subtle facial expression"),
        "clothing_state": facts.get("clothing_state", ""),
        "audio": {
            "narration": [],
            "dialogue": [],
            "sound_effects": ["ambient room presence"],
            "music": {
                "mood": "sensual" if act in ("kiss", "touch", "oral", "vaginal", "anal") else "cinematic",
                "intensity": pace,
            },
        },
        "continuity": {},
    }
    return compiled


def enforce_wardrobe(compiled_scenes: List[Dict[str, Any]]) -> None:
    """
    Enforces that any scene with an intimate exposed act automatically has
    a lower-body removal event if not previously triggered, guaranteeing correct
    genital references and Stage B activation.
    """
    seen_unclothed = set()
    for s in compiled_scenes:
        if s.get("act") in EXPOSED_ACTS:
            for cid in s.get("character_ids", []):
                already_removed = any(
                    (
                        (e.get("character_id") == cid and e.get("event") in ("remove_bottom", "remove_all"))
                        if isinstance(e, dict)
                        else (str(e) in ("remove_bottom", "remove_all"))
                    )
                    for sc in compiled_scenes
                    if sc.get("order", 0) <= s.get("order", 0)
                    for e in sc.get("wardrobe_events", [])
                )
                if cid not in seen_unclothed and not already_removed:
                    s.setdefault("wardrobe_events", []).append(
                        {"character_id": cid, "event": "remove_bottom"}
                    )
                seen_unclothed.add(cid)
