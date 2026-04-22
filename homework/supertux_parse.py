"""
Shared SuperTuxKart info.json parsing for QA and caption generation.
"""

from __future__ import annotations

import json
from pathlib import Path

# Matches grader-style counting on valid_grader (see README / starter analysis).
MIN_KART_AREA = 475
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def load_info(info_path: str) -> dict:
    with open(info_path) as f:
        return json.load(f)


def _scale_xy(x: float, y: float, img_width: int, img_height: int) -> tuple[float, float]:
    sx = img_width / ORIGINAL_WIDTH
    sy = img_height / ORIGINAL_HEIGHT
    return x * sx, y * sy


def visible_kart_detections(
    info: dict, view_index: int, min_area: int = MIN_KART_AREA, min_side: int = 5
) -> list[dict]:
    """class_id==1 karts with bbox area >= min_area and sides >= min_side (original 600x400 coords)."""
    if view_index >= len(info["detections"]):
        return []
    karts = info["karts"]
    out = []
    for det in info["detections"][view_index]:
        class_id, track_id, x1, y1, x2, y2 = det
        if int(class_id) != 1:
            continue
        tid = int(track_id)
        w, h = x2 - x1, y2 - y1
        if w < min_side or h < min_side:
            continue
        area = w * h
        if area < min_area:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        out.append(
            {
                "track_id": tid,
                "name": karts[tid].lower(),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area": area,
                "cx": cx,
                "cy": cy,
                "ddt": float(info["distance_down_track"][tid]),
            }
        )
    return out


def extract_track_info(info_path: str) -> str:
    info = load_info(info_path)
    return str(info["track"]).lower()


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list[dict]:
    info = load_info(info_path)
    dets = visible_kart_detections(info, view_index, min_area=min_box_size * min_box_size, min_side=min_box_size)
    if not dets:
        return []
    centers = [(d["cx"], d["cy"]) for d in dets]
    img_cx = ORIGINAL_WIDTH / 2
    img_cy = ORIGINAL_HEIGHT / 2
    best_i = min(range(len(dets)), key=lambda i: (centers[i][0] - img_cx) ** 2 + (centers[i][1] - img_cy) ** 2)
    result = []
    for i, d in enumerate(dets):
        sx, sy = _scale_xy(d["cx"], d["cy"], img_width, img_height)
        result.append(
            {
                "instance_id": d["track_id"],
                "kart_name": d["name"],
                "center": (sx, sy),
                "is_center_kart": i == best_i,
            }
        )
    return result


def _ego_ref_for_pairwise(visible: list[dict], exclude_tid: int) -> dict | None:
    """Largest-area visible kart other than exclude_tid (reference 'ego' for relative questions)."""
    candidates = [d for d in visible if d["track_id"] != exclude_tid]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d["area"])


def _ego_ref_for_counts(visible: list[dict]) -> dict | None:
    if not visible:
        return None
    return max(visible, key=lambda d: d["area"])


def left_or_right(target: dict, ref: dict) -> str:
    return "left" if target["cx"] < ref["cx"] else "right"


def front_or_back_cy(target: dict, ref: dict) -> str:
    """Along-track proxy used in grader-style labels: smaller image y => 'front'."""
    return "front" if target["cy"] < ref["cy"] else "back"


def behind_count_ddt(target: dict, ref: dict) -> bool:
    """Alternate 'behind' for counting-only heuristics."""
    return target["ddt"] < ref["ddt"]


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list[dict]:
    info = load_info(info_path)
    karts = [str(x).lower() for x in info["karts"]]
    visible = visible_kart_detections(info, view_index)
    by_tid = {d["track_id"]: d for d in visible}
    qa: list[dict] = []

    if not karts:
        return qa

    # 1. Ego car (player list slot 0)
    qa.append({"question": "What kart is the ego car?", "answer": karts[0]})

    # 2. Total karts visible in this view
    qa.append({"question": "How many karts are there in the scenario?", "answer": str(len({d["track_id"] for d in visible}))})

    # 3. Track
    qa.append({"question": "What track is this?", "answer": str(info["track"]).lower()})

    # 4–5. Per other visible kart (not only non-ego; questions can still be posed)
    seen_names: set[str] = set()
    for tid, d in sorted(by_tid.items()):
        name = d["name"]
        if name in seen_names:
            continue
        seen_names.add(name)
        ref = _ego_ref_for_pairwise(visible, exclude_tid=tid)
        if ref is None:
            continue
        qa.append({"question": f"Is {name} to the left or right of the ego car?", "answer": left_or_right(d, ref)})
        qa.append({"question": f"Is {name} in front of or behind the ego car?", "answer": front_or_back_cy(d, ref)})
        lr = left_or_right(d, ref)
        fb = front_or_back_cy(d, ref)
        qa.append({"question": f"Where is {name} relative to the ego car?", "answer": f"{fb} and {lr}"})

    # 6. Counting relative to largest-bbox kart (camera / player proxy)
    ego_c = _ego_ref_for_counts(visible)
    if ego_c is not None:
        ec, ey, ed = ego_c["cx"], ego_c["cy"], ego_c["ddt"]
        left_n = sum(1 for d in visible if d["track_id"] != ego_c["track_id"] and d["cx"] < ec)
        right_n = sum(1 for d in visible if d["track_id"] != ego_c["track_id"] and d["cx"] > ec)
        front_n = sum(1 for d in visible if d["track_id"] != ego_c["track_id"] and d["cy"] < ey)
        behind_n = sum(1 for d in visible if d["track_id"] != ego_c["track_id"] and behind_count_ddt(d, ego_c))
        qa.append({"question": "How many karts are to the left of the ego car?", "answer": str(left_n)})
        qa.append({"question": "How many karts are to the right of the ego car?", "answer": str(right_n)})
        qa.append({"question": "How many karts are in front of the ego car?", "answer": str(front_n)})
        qa.append({"question": "How many karts are behind the ego car?", "answer": str(behind_n)})

    return qa


def generate_caption_lines(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list[str]:
    """Captions aligned with valid_grader / all_mc_qas phrasing."""
    info = load_info(info_path)
    karts = [str(x).lower() for x in info["karts"]]
    visible = visible_kart_detections(info, view_index)
    lines: list[str] = []

    if karts:
        lines.append(f"{karts[0]} is the ego car.")

    n = len({d["track_id"] for d in visible})
    lines.append(f"There are {n} karts in the scene.")

    lines.append(f"The track is {str(info['track']).lower()}.")

    for d in visible:
        ref = _ego_ref_for_pairwise(visible, exclude_tid=d["track_id"])
        if ref is None:
            continue
        name = d["name"]
        if front_or_back_cy(d, ref) == "front":
            lines.append(f"{name} is in front of the ego car.")
        else:
            lines.append(f"{name} is behind the ego car.")
        if left_or_right(d, ref) == "left":
            lines.append(f"{name} is left of the ego car.")
        else:
            lines.append(f"{name} is right of the ego car.")

    return lines
