import re
from typing import Dict, Any, List, Tuple

_HDR_DUR = re.compile(r"(\d+(?:\.\d+)?)\s*s\b", re.I)
_PATTERNS = [
    re.compile(r"^\s*Scene\s+(\d+)\s*(?:\[([^\]]*)\])?\s*[:.\-]?\s*$", re.I | re.M),  # header on own line
    re.compile(r"^\s*Scene\s+(\d+)\s*(?:\[([^\]]*)\])?\s*[:.\-]\s*", re.I | re.M),    # "Scene 3: text"
    re.compile(r"^\s*(\d+)[.)]\s+()", re.M),                                           # "1. text"
]


def parse_story(
    story: str,
    target_scenes: int | None = None,
    default_dur: float = 5.0,
) -> Tuple[Dict[str, str], str, List[Dict[str, Any]]]:
    """
    Parses any story text format (scene headers with brackets/durations,
    numbered lists, plain paragraphs, or continuous prose) into structured scene items.
    """
    story = story.strip()
    scenes: List[Dict[str, Any]] = []
    preamble = ""

    for pat in _PATTERNS:
        ms = list(pat.finditer(story))
        if len(ms) >= 2:
            preamble = story[:ms[0].start()]
            for i, m in enumerate(ms):
                end = ms[i + 1].start() if i + 1 < len(ms) else len(story)
                hdr = m.group(2) or ""
                d = _HDR_DUR.search(hdr)
                shot_hint = _HDR_DUR.sub("", hdr).strip(" ,")
                scenes.append({
                    "order": i + 1,
                    "shot_hint": shot_hint,
                    "duration": float(d.group(1)) if d else default_dur,
                    "text": " ".join(story[m.end():end].split()),
                })
            break

    if not scenes:  # unstructured story: paragraphs, then sentence groups
        paras = [p.strip() for p in re.split(r"\n\s*\n", story) if p.strip()]
        if target_scenes and len(paras) != target_scenes:
            sents = re.split(r"(?<=[.!?])\s+", " ".join(paras))
            n = target_scenes
            size = max(1, round(len(sents) / n))
            paras = [" ".join(sents[i:i + size]) for i in range(0, len(sents), size)][:n]
        scenes = [
            {
                "order": i + 1,
                "shot_hint": "",
                "duration": default_dur,
                "text": " ".join(p.split()),
            }
            for i, p in enumerate(paras)
        ]

    meta = dict(re.findall(r"^\s*(Title|Setting)\s*:\s*(.+)$", preamble, re.I | re.M))
    return {k.lower(): v.strip() for k, v in meta.items()}, preamble.strip(), scenes

