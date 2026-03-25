import json
import sys

from dreamsApp.app.utils.memory_analyser import analyse_memory


def _bar(value):
    pct = int(round(float(value) * 100))
    return pct, "█" * max(1, pct // 5) if pct > 0 else ""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_pipeline.py path/to/image.jpg")
        raise SystemExit(1)

    res = analyse_memory(sys.argv[1])
    em, sc = res["emotion"], res["scene"]
    h, hb = _bar(em.get("happy", 0.0))
    s, sb = _bar(em.get("sad", 0.0))
    n, nb = _bar(em.get("neutral", 0.0))
    top3 = " | ".join([f"{x.get('label')} ({int(round(float(x.get('confidence', 0)) * 100))}%)" for x in sc.get("scene_raw_top3", [])]) or "none"

    print("EMOTION ANALYSIS")
    print(f"Dominant: {em.get('dominant_emotion', 'unknown')}")
    print(f"Happy:   {h:>2}%  {hb}")
    print(f"Sad:     {s:>2}%  {sb}")
    print(f"Neutral: {n:>2}%  {nb}\n")
    print("SCENE ANALYSIS")
    print(f"Type:       {sc.get('scene_type', 'unknown')}")
    print(f"Confidence: {int(round(float(sc.get('scene_confidence', 0.0)) * 100))}%")
    print(f"Top 3:      {top3}\n")
    print("COMBINED INSIGHT")
    print(res.get("combined_insight", ""))
    print("\nRAW JSON")
    print(json.dumps(res, indent=2))