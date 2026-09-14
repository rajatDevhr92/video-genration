from .llm_provider import LLMProvider


class MockLLMProvider(LLMProvider):

    def generate_json(self, prompt: str):

        return {
            "title": "The Rainy Detective",
            "summary": (
                "A detective walks through a futuristic city "
                "during a storm while searching for a mysterious clue."
            ),

            "characters": [
                {
                    "id": "detective_001",
                    "name": "Detective",
                    "appearance": (
                        "35-year-old man with short black hair, "
                        "trimmed beard and serious expression"
                    ),
                    "clothing": (
                        "dark blue detective trench coat, "
                        "black trousers and black leather shoes"
                    ),
                    "personality": "serious and observant",
                    "visual_anchor": (
                        "35-year-old detective, short black hair, "
                        "trimmed beard, dark blue trench coat"
                    ),
                }
            ],

            "locations": [
                {
                    "id": "city_001",
                    "name": "Futuristic City",
                    "description": (
                        "A dense futuristic city with enormous "
                        "skyscrapers, wet streets and neon signs."
                    ),
                    "visual_style": (
                        "dark cinematic science-fiction environment, "
                        "neon reflections and heavy rain"
                    ),
                }
            ],

            "scenes": [
                {
                    "id": "scene_001",
                    "order": 1,
                    "duration": 5,
                    "location_id": "city_001",
                    "character_ids": ["detective_001"],
                    "description": (
                        "The detective walks alone through the "
                        "rainy futuristic city."
                    ),
                    "action": (
                        "The detective slowly walks forward "
                        "through heavy rain."
                    ),
                    "camera": (
                        "slow cinematic tracking shot moving "
                        "backward in front of the detective"
                    ),
                    "lighting": (
                        "neon city lights reflecting on wet pavement"
                    ),
                    "video_prompt": (
                        "A 35-year-old detective with short black hair "
                        "and a trimmed beard, wearing a dark blue "
                        "detective trench coat, walks slowly through "
                        "a futuristic city at midnight during heavy rain. "
                        "Neon signs reflect on the wet pavement. "
                        "The camera slowly tracks backward in front of him. "
                        "Realistic human movement, photorealistic cinematic "
                        "style, atmospheric lighting, detailed environment."
                    ),
                }
            ],
        }