from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class MotionType(str, Enum):
    idle = "idle"
    walk = "walk"
    sway = "sway"
    turn = "turn"
    sit_down = "sit_down"
    gesture = "gesture"
    thrust = "thrust"
    grind = "grind"
    rock = "rock"
    bounce = "bounce"
    circle = "circle"
    pulse = "pulse"


class Amplitude(str, Enum):
    micro = "micro"
    small = "small"
    medium = "medium"
    large = "large"


class Easing(str, Enum):
    linear = "linear"
    ease_in_out = "ease_in_out"
    ease_in = "ease_in"
    ease_out = "ease_out"


@dataclass
class Waypoint:
    t: float
    pos: Tuple[float, float]
    facing_deg: float = 0.0
    head_yaw_deg: float = 0.0


@dataclass
class CharacterMotion:
    character_id: str
    motion_type: MotionType = MotionType.thrust
    amplitude: Amplitude = Amplitude.medium
    max_speed_m_s: float = 0.45
    hands: str = "anchored"
    easing: Easing = Easing.ease_in_out
    start_pose_ref: Optional[str] = None
    end_pose_ref: Optional[str] = None
    path: List[Waypoint] = field(default_factory=list)

    # Intimate / adult kinetic controls
    hip_motion: Optional[str] = None       # e.g., "forward-back arc", "circular grinding"
    pelvis_tilt: Optional[str] = None      # e.g., "posterior tilt", "neutral"
    depth_change: Optional[str] = None     # e.g., "constant", "deepening", "pulsing"
    contact_pressure: Optional[str] = None # e.g., "firm", "light", "heavy"


@dataclass
class Relation:
    a: str
    b: str
    distance_start_m: float = 0.10
    distance_end_m: float = 0.10
    a_faces_b: bool = True
    b_faces_a: bool = True


@dataclass
class Motion:
    characters: List[CharacterMotion] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    bpm: Optional[float] = 72.0
    beat_aligned: bool = True
    control_type: str = "pose_depth"


def motion_to_prompt(motion: Optional[Any]) -> str:
    """
    Auto-generates a highly detailed natural language video motion prompt
    for Wan 2.2 from the structured Motion dataclass or dict representation.
    """
    if not motion:
        return "Natural, continuous, realistic physical motion."

    if isinstance(motion, dict):
        bpm = motion.get("bpm")
        characters = motion.get("characters", [])
        relations = motion.get("relations", [])
    else:
        bpm = getattr(motion, "bpm", None)
        characters = getattr(motion, "characters", [])
        relations = getattr(motion, "relations", [])

    if not characters:
        return "Natural, continuous, realistic physical motion."

    prompt_parts = []
    if bpm:
        prompt_parts.append(f"Rhythmic cadence locked at {bpm} BPM.")

    for cm in characters:
        if isinstance(cm, dict):
            c_id = cm.get("character_id", "")
            m_type = cm.get("motion_type", "thrust")
            if hasattr(m_type, "value"):
                m_type = m_type.value
            amp = cm.get("amplitude", "medium")
            if hasattr(amp, "value"):
                amp = amp.value
            speed = cm.get("max_speed_m_s", 0.45)
            hands = cm.get("hands", "anchored")
            hip_motion = cm.get("hip_motion")
            pelvis_tilt = cm.get("pelvis_tilt")
            depth_change = cm.get("depth_change")
            contact_pressure = cm.get("contact_pressure")
        else:
            c_id = getattr(cm, "character_id", "")
            m_type = getattr(cm, "motion_type", "thrust")
            if hasattr(m_type, "value"):
                m_type = m_type.value
            amp = getattr(cm, "amplitude", "medium")
            if hasattr(amp, "value"):
                amp = amp.value
            speed = getattr(cm, "max_speed_m_s", 0.45)
            hands = getattr(cm, "hands", "anchored")
            hip_motion = getattr(cm, "hip_motion", None)
            pelvis_tilt = getattr(cm, "pelvis_tilt", None)
            depth_change = getattr(cm, "depth_change", None)
            contact_pressure = getattr(cm, "contact_pressure", None)

        c_desc = f"Character '{c_id}': {m_type} motion with {amp} amplitude at speed {speed} m/s. Hands: {hands}."
        if hip_motion:
            c_desc += f" Hip motion: {hip_motion}."
        if pelvis_tilt:
            c_desc += f" Pelvis tilt: {pelvis_tilt}."
        if depth_change:
            c_desc += f" Penetration depth: {depth_change}."
        if contact_pressure:
            c_desc += f" Contact pressure: {contact_pressure}."

        prompt_parts.append(c_desc)

    for rel in relations:
        if isinstance(rel, dict):
            rel_a = rel.get("a", "")
            rel_b = rel.get("b", "")
            d_start = rel.get("distance_start_m", 0.10)
            d_end = rel.get("distance_end_m", 0.10)
        else:
            rel_a = getattr(rel, "a", "")
            rel_b = getattr(rel, "b", "")
            d_start = getattr(rel, "distance_start_m", 0.10)
            d_end = getattr(rel, "distance_end_m", 0.10)

        prompt_parts.append(
            f"Spatial relation between {rel_a} and {rel_b}: maintain constant close physical proximity "
            f"({d_start}m to {d_end}m distance)."
        )

    return " ".join(prompt_parts)


@dataclass
class Character:
    id: str
    name: str
    appearance: str
    clothing: str
    personality: str = ""
    visual_anchor: str = ""


@dataclass
class Location:
    id: str
    name: str
    description: str
    visual_style: str = ""


@dataclass
class PersistentObject:
    id: str
    name: str
    state: str
    introduced_in_scene: int = 0


@dataclass
class Continuity:
    persistent_objects: List[Dict[str, Any]] = field(
        default_factory=list
    )
    notes: List[str] = field(
        default_factory=list
    )


@dataclass
class Scene:
    id: str
    order: int
    duration: float
    location_id: str
    character_ids: List[str]
    description: str
    action: str
    camera: str
    lighting: str
    video_prompt: str
    clothing_state: str = ""
    pose_type: str = ""
    penetration_state: str = ""
    primary_motion: str = ""
    previous_scene_context: str = ""
    pose_lock: str = ""
    hand_positions: str = ""
    genital_alignment: str = ""
    contact_zone: str = ""
    motion_intensity: str = ""
    expression: str = ""
    motion: Optional[Motion] = None
    continuity: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class StoryPlan:
    title: str
    summary: str
    characters: List[Character] = field(
        default_factory=list
    )
    locations: List[Location] = field(
        default_factory=list
    )
    scenes: List[Scene] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,

            "characters": [
                {
                    "id": c.id,
                    "name": c.name,
                    "appearance": c.appearance,
                    "clothing": c.clothing,
                    "personality": c.personality,
                    "visual_anchor": c.visual_anchor,
                }
                for c in self.characters
            ],

            "locations": [
                {
                    "id": l.id,
                    "name": l.name,
                    "description": l.description,
                    "visual_style": l.visual_style,
                }
                for l in self.locations
            ],
            "scenes": [
                {
                    "id": s.id,
                    "order": s.order,
                    "duration": s.duration,
                    "location_id": s.location_id,
                    "character_ids": s.character_ids,
                    "clothing_state": getattr(s, "clothing_state", ""),
                    "pose_type": getattr(s, "pose_type", ""),
                    "penetration_state": getattr(s, "penetration_state", ""),
                    "primary_motion": getattr(s, "primary_motion", ""),
                    "previous_scene_context": getattr(s, "previous_scene_context", ""),
                    "pose_lock": getattr(s, "pose_lock", ""),
                    "hand_positions": getattr(s, "hand_positions", ""),
                    "genital_alignment": getattr(s, "genital_alignment", ""),
                    "contact_zone": getattr(s, "contact_zone", ""),
                    "motion_intensity": getattr(s, "motion_intensity", ""),
                    "expression": getattr(s, "expression", ""),
                    "motion": getattr(s, "motion", None),
                    "description": s.description,
                    "action": s.action,
                    "camera": s.camera,
                    "lighting": s.lighting,
                    "video_prompt": s.video_prompt,
                    "continuity": s.continuity,
                }
                for s in self.scenes
            ],
        }


StoryPlanSchema = StoryPlan
