import tempfile

from PIL import Image

from dreamsApp.app.utils.memory_analyser import analyse_memory


def test_analyse_memory_pipeline():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        Image.new("RGB", (128, 128), color=(120, 180, 200)).save(tmp.name)
        result = analyse_memory(tmp.name)

    print("analysis result:", result)

    assert set(result.keys()) == {"emotion", "scene", "combined_insight"}
    assert isinstance(result["emotion"]["dominant_emotion"], str)
    assert result["emotion"]["dominant_emotion"].strip() != ""
    assert result["scene"]["scene_type"] in {
        "clinical_or_institutional",
        "faith_community",
        "recovery_support",
        "residential_or_transitional",
        "shelter_or_dropin",
        "outdoor_or_wilderness",
        "unknown",
    }
    assert isinstance(result["combined_insight"], str)
    assert result["combined_insight"].strip() != ""
