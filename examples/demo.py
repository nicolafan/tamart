"""Qualitative TAM demo using the TAMExplainer API.

Run from the repo root:
    python examples/demo.py
"""
from tamart.tam import TAMExplainer


def main():
    explainer = TAMExplainer(model_name="Qwen/Qwen2-VL-2B-Instruct")

    img = "imgs/demo.jpg"
    prompt = "Describe this image."
    result = explainer.explain(img, prompt, save_dir="imgs/vis_img")
    print("Image caption:", result["text"])

    # video demo: Qwen merges next frames, so we repeat each frame twice
    frames = []
    for i in range(10):
        frames.extend([f"imgs/frames/{str(i).zfill(4)}.jpg"] * 2)
    result = explainer.explain(frames, "Describe this video.", save_dir="imgs/vis_video")
    print("Video caption:", result["text"])


if __name__ == "__main__":
    main()
