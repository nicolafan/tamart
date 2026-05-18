# tamart

Wrapper project around **Token Activation Map (TAM)** ([ICCV 2025 Oral](https://arxiv.org/abs/2506.23270)) — a method to produce clear visualizations of the visual evidence behind every word a Multimodal LLM generates.

The original TAM code is packaged as `tamart.tam` with a single high-level entry point: the `TAMExplainer` class.

![Overview](imgs/overview.jpg)

## Install

```bash
pip install -e .
# for text visualization:
sudo apt-get install texlive-xetex
```

## Quick start

```python
from tamart.tam import TAMExplainer

# loads the MLLM once
explainer = TAMExplainer(model_name="Qwen/Qwen2-VL-2B-Instruct")

# single image — returns generated text + per-token activation maps
result = explainer.explain("imgs/demo.jpg", "Describe this image.")
print(result["text"])           # generated answer
print(len(result["maps"]))      # one map per generated token

# save TAM visualizations as JPGs (one per generation round)
explainer.explain("imgs/demo.jpg", "Describe this image.", save_dir="out/")

# video: pass a list of frame paths (Qwen merges next frames, so duplicate each)
frames = [f"imgs/frames/{str(i).zfill(4)}.jpg" for i in range(10) for _ in range(2)]
explainer.explain(frames, "Describe this video.", save_dir="out_video/")
```

The result dict contains:
- `text` (str) — the generated answer.
- `maps` (list[np.ndarray]) — one TAM activation map per generated token.
- `tokens` (list[int]) — generated token ids (prompt trimmed).

## Model cache location (`HF_HOME`)

Importing `tamart` (or any submodule) automatically sets `HF_HOME` so all downloaded weights and Hugging Face caches land under `data/hf/` at the repo root. The default lives in `.env` at the repo root — edit it to change the path:

```
HF_HOME=data/hf
```

Relative paths in `.env` are resolved against the repo root, and `~` is expanded, so the location is stable no matter where you run scripts from. `.env` overrides any shell-level `HF_HOME` — if you want a different cache for a particular run, edit `.env`.

**Notebooks:** put `import tamart` at the top *before* any `import transformers` / `from huggingface_hub import ...`. HF libraries read `HF_HOME` at their own import time, so the bootstrap has to run first.

## Supported MLLMs

Currently only **Qwen2-VL** (any size). LLaVA and InternVL3 are upstream-supported by the underlying `TAM` function but not wired into `TAMExplainer` yet.

## Examples

```bash
# qualitative demo
python examples/demo.py

# quantitative eval on a dataset (COCO / GranDf / OpenPSG layout supported)
python examples/eval.py Qwen/Qwen2-VL-2B-Instruct data/coco2014
# add a vis_path to also save TAM images:
python examples/eval.py Qwen/Qwen2-VL-2B-Instruct data/coco2014 out/
```

Download the formatted datasets at [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ylini_connect_ust_hk/EXL-stkCxk5DnwRkNw9MgSABu1vFPv_0FI60yxl0OYxSGQ?e=V3qjHh) or [Hugging Face](https://huggingface.co/datasets/yili7eli/TAM/tree/main).

## Layout

```
src/tamart/tam/
  core.py        — the TAM algorithm (rank Gaussian filter, ECI, latex vis)
  explainer.py   — TAMExplainer class (Qwen2-VL wrapper)
  eval.py        — IoU / ROUGE-L / METEOR metrics over TAM maps
  qwen_utils.py  — Qwen vision pre-processing helpers
examples/
  demo.py        — qualitative example
  eval.py        — quantitative eval script
```

## Using `TAM` directly

If you need to wire up another MLLM, the raw `TAM` function is still exposed:

```python
from tamart.tam import TAM
```

See [`src/tamart/tam/core.py`](src/tamart/tam/core.py) for the full signature and [`src/tamart/tam/explainer.py`](src/tamart/tam/explainer.py) for an end-to-end example of how to wire a model in.

## License

MIT.

## Citation

```
@InProceedings{Li_2025_ICCV,
    author    = {Li, Yi and Wang, Hualiang and Ding, Xinpeng and Wang, Haonan and Li, Xiaomeng},
    title     = {Token Activation Map to Visually Explain Multimodal LLMs},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {48-58}
}
```
