from dataclasses import dataclass, field
from typing import List, Dict, Any


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

