import os

from background_place_detector import BackgroundPlaceDetector

_DETECTOR = None

_CATEGORY_KEYWORDS = {
    "clinical_or_institutional": ["hospital", "clinic", "laboratory", "waiting_room", "classroom", "office"],
    "faith_community": ["church", "mosque", "temple", "synagogue", "abbey", "cathedral", "chapel"],
    "recovery_support": ["cafeteria", "restaurant", "library", "community_center", "meeting_room"],
    "residential_or_transitional": ["bedroom", "living_room", "kitchen", "home", "house", "apartment"],
    "shelter_or_dropin": ["dorm", "hostel", "shelter", "reception", "lobby"],
    "outdoor_or_wilderness": ["park", "forest", "beach", "mountain", "trail", "field", "garden", "outdoor", "street"],
}


def _map_scene(place_label):
    label = (place_label or "").lower()
    for scene_type, keys in _CATEGORY_KEYWORDS.items():
        if any(k in label for k in keys):
            return scene_type
    return "outdoor_or_wilderness" if "/outdoor" in label else "clinical_or_institutional"


def classify_scene(image_path):
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = BackgroundPlaceDetector(arch="resnet18")
    top = _DETECTOR.predict_place(image_path, top_k=3)
    if not top:
        return {"scene_type": "unknown", "scene_confidence": 0.0, "scene_raw_top3": []}
    primary_conf, primary_label = top[0]
    return {
        "scene_type": _map_scene(primary_label),
        "scene_confidence": float(primary_conf),
        "scene_raw_top3": [{"label": lbl, "confidence": float(conf)} for conf, lbl in top],
    }