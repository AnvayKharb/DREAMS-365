# Combined analysis pipeline — fine-tuning and CLIP secondary verification planned for GSoC coding period
import logging
import os
import sys

logger = logging.getLogger(__name__)
_SCENE_CLASSIFIER = None
_EMOTION_COMPONENTS = None


def _default_scene():
    return {"scene_type": "unknown", "scene_confidence": 0.0, "scene_raw_top3": []}


def _default_emotion():
    return {"dominant_emotion": "unknown", "happy": 0.0, "sad": 0.0, "neutral": 0.0}


def _humanize_scene(scene_type):
    return (scene_type or "unknown").replace("_", " ").replace("/", " or ")


def _load_scene_classifier():
    global _SCENE_CLASSIFIER
    if _SCENE_CLASSIFIER is None:
        from dreamsApp.app.utils.places365_classifier import classify_scene
        _SCENE_CLASSIFIER = classify_scene
    return _SCENE_CLASSIFIER


def _load_emotion_components():
    global _EMOTION_COMPONENTS
    if _EMOTION_COMPONENTS is not None:
        return _EMOTION_COMPONENTS
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    face_root = os.path.join(base, "face-classification", "face_classification")
    src = os.path.join(face_root, "src")
    if src not in sys.path:
        sys.path.append(src)
    import cv2
    from keras.models import load_model
    labels = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral",
    }

    detector = cv2.CascadeClassifier(
        os.path.join(face_root, "trained_models", "detection_models", "haarcascade_frontalface_default.xml")
    )
    model = load_model(
        os.path.join(face_root, "trained_models", "emotion_models", "fer2013_mini_XCEPTION.102-0.66.hdf5"),
        compile=False,
    )
    label_to_idx = {v: k for k, v in labels.items()}
    _EMOTION_COMPONENTS = {
        "cv2": cv2,
        "model": model,
        "labels": labels,
        "label_to_idx": label_to_idx,
        "detector": detector,
        "target_size": model.input_shape[1:3],
    }
    return _EMOTION_COMPONENTS


def preload_models():
    """Preload both scene and emotion models so first request is fast."""
    _load_scene_classifier()
    _load_emotion_components()


def _infer_emotion_scores(image_path):
    import numpy as np

    def _preprocess_input(x):
        x = x.astype("float32")
        x = x / 255.0
        x = x - 0.5
        return x * 2.0

    c = _load_emotion_components()
    cv2 = c["cv2"]
    model = c["model"]
    labels = c["labels"]
    label_to_idx = c["label_to_idx"]
    detector = c["detector"]
    image = cv2.imread(image_path)
    if image is None:
        return _default_emotion()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try multiple detector settings and contrast-normalized grayscale
    faces = ()
    for source in (gray, cv2.equalizeHist(gray)):
        for sf, mn in ((1.3, 5), (1.2, 4), (1.1, 3)):
            faces = detector.detectMultiScale(source, sf, mn)
            if len(faces) > 0:
                break
        if len(faces) > 0:
            break

    if len(faces) == 0:
        return _default_emotion()

    # Pick the largest detected face for stable emotion output.
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    h_img, w_img = gray.shape[:2]
    x1, x2, y1, y2 = x, x + w, y, y + h
    x1 = max(0, min(x1, w_img - 1))
    x2 = max(1, min(x2, w_img))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(1, min(y2, h_img))
    if x2 <= x1 or y2 <= y1:
        return _default_emotion()

    face = cv2.resize(gray[y1:y2, x1:x2], c["target_size"])
    face = _preprocess_input(face)
    face = np.expand_dims(np.expand_dims(face, 0), -1)
    probs = model.predict(face, verbose=0)[0].astype(float)
    dominant = labels[int(probs.argmax())]
    return {
        "dominant_emotion": dominant,
        "happy": float(probs[label_to_idx.get("happy", 3)]),
        "sad": float(probs[label_to_idx.get("sad", 4)]),
        "neutral": float(probs[label_to_idx.get("neutral", 6)]),
    }


def analyse_memory(image_path):
    """Runs both Places365 and face emotion analysis and returns combined results."""
    scene = _default_scene()
    emotion = _default_emotion()

    try:
        logger.info("Running scene classifier for %s", image_path)
        raw_scene = _load_scene_classifier()(image_path) or {}
        scene = {
            "scene_type": raw_scene.get("scene_type", "unknown"),
            "scene_confidence": float(raw_scene.get("scene_confidence", 0.0) or 0.0),
            "scene_raw_top3": raw_scene.get("scene_raw_top3", []),
        }
    except Exception as err:
        logger.exception("Scene classifier failed: %s", err)

    try:
        logger.info("Running face emotion classifier for %s", image_path)
        _load_emotion_components()
        emotion = _infer_emotion_scores(image_path)
    except Exception as err:
        logger.exception("Emotion classifier failed: %s", err)

    insight = f"Person appears {emotion['dominant_emotion']} in an {_humanize_scene(scene['scene_type'])} setting"
    return {"emotion": emotion, "scene": scene, "combined_insight": insight}


try:
    preload_models()
except Exception as preload_err:
    logger.warning("Model preload incomplete; lazy fallback enabled: %s", preload_err)
