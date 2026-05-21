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

## Datasets

### WikiArt — top 1000 most-viewed paintings

Pulls the ranked most-viewed list from the WikiArt v2 JSON API (`/en/api/2/MostViewedPaintings`, paginated), then downloads each painting's full metadata and image at the highest resolution available (falling back through `!HD` / `!HalfHD` / `!Large` variants if the original is missing).

```bash
# from the repo root
python -m tamart.data.download data/datasets/wikiart_most_viewed -n 1000
```

Output layout:

```
data/datasets/wikiart_most_viewed/
  images/0001_<slug>.jpg, 0002_<slug>.jpg, ...     # ranked, highest res available
  annotations.json                                  # ordered metadata (title, artist, year, styles, genres, description, ...)
```

`annotations.json` is rewritten incrementally after each painting, so a crash mid-run doesn't lose progress. Re-running skips images already on disk (pass `--overwrite` to force re-download); listing and metadata calls are repeated each run.

#### Annotate with a multimodal LLM

Probe a Qwen2-VL model with two prompts per image — one asking for `{title, artist}` and one for `{artist}` — and store the raw JSON responses next to the dataset:

```bash
python -m tamart.data.annotate
# or override the defaults:
python -m tamart.data.annotate --model-name Qwen/Qwen2-VL-7B-Instruct --max-new-tokens 32
```

Results land at `<data-dir>/<model-name>/results.json` (the model name's `/` becomes a nested folder, so the default writes to `wikiart_most_viewed/Qwen/Qwen2-VL-2B-Instruct/results.json`). The file is rewritten atomically after every image, and re-running resumes from the first image that doesn't yet have both prompts answered.

#### Grade the annotations

Parse those raw responses into `{title, artist}` / `{artist}` objects and use a small Qwen3 Instruct LLM as a judge against the WikiArt ground truth:

```bash
python -m tamart.data.check
# explicit:
python -m tamart.data.check --source-model Qwen/Qwen2-VL-2B-Instruct \
                            --judge-model Qwen/Qwen3-4B-Instruct-2507
```

Writes `<data-dir>/<source-model>/results_clean.json`. For each image:

- `title_artist` → `{title, artist, result}` with `result ∈ {both_correct, title_only, artist_only, both_wrong}`
- `artist` → `{artist, result}` with `result ∈ {correct, wrong}`

Responses that don't contain valid JSON are recorded as `{"result": "parse_error"}` and skipped by the judge. The judge is the **non-thinking** Qwen3 Instruct variant (the 2507 release) — the task is short structured classification, so the thinking variant would be overkill. The script is resumable: rows that already carry a non-`parse_error` result are not re-judged.

## Method

### Precompute TAM maps

Run the artist-JSON prompt through Qwen2-VL on every painting, identify the generated tokens covering the artist name, and save their merged TAM saliency as a safetensors file per image. The map is float32 at the model's native patch grid, normalized to `[0, 1]` (the max often sits below 1.0 because TAM's min-max is taken jointly over image+text tokens, so the image's max needn't hit the top).

```bash
# default: merge maps from every token overlapping the artist name
python -m tamart.method.compute
# or only the first artist-name token's map:
python -m tamart.method.compute --token-mode first
# smoke test:
python -m tamart.method.compute --limit 1
```

Output layout (mirrors `tamart.data.annotate`):

```
<data-dir>/<model-name>/maps/<image_stem>.safetensors
```

Each safetensors file holds one tensor named `map` plus string metadata: `model`, `prompt`, `text` (raw answer), `artist` (parsed), `token_mode`, `artist_token_indices` (every token overlapping the name), and `selected_token_indices` (the subset actually used). The script is resumable — images that already have a maps file are skipped on re-run. Images whose answer doesn't parse, or whose artist name doesn't align to any generated token, are skipped with a warning.

Read a map back with:

```python
from safetensors import safe_open

with safe_open("path/to/0001_mona-lisa.safetensors", framework="numpy") as f:
    tam_map = f.get_tensor("map")     # float32, [H_patch, W_patch], values in [0, 1]
    meta    = f.metadata()            # str→str dict; lists are JSON-encoded
```

## Experiments

### Describe — caption every painting with TAM maps

For every image in `<data-dir>/images`, run Qwen2-VL with a single "describe the content and style" prompt and save the generated caption, the per-token TAM activation maps, and the processed image. The saved layout is exactly what `TAMExplainer.explain_interactive_precomputed` reads back, so a notebook can replay any annotation without re-running inference.

```bash
python -m tamart.experiments.describe --batch-size 4
# or override the defaults:
python -m tamart.experiments.describe --model-name Qwen/Qwen2-VL-7B-Instruct --max-new-tokens 256
```

Output layout (one subfolder per painting):

```
<data-dir>/describe/<image_stem>/
  answer.json          # text, generated token ids, token_labels, per-token img/txt stats
  maps.safetensors     # one float32 map per generated token, keyed by token index
  proc_img.png         # the processed RGB image TAM overlays maps onto
```

The script is resumable — pass `--no-overwrite` (the default) to skip paintings that already have an `answer.json`.

### Classify — group the description tokens into categories

Feed each `describe/<stem>/answer.json` caption — presented as a numbered token list, where a leading `·` marks the start of a new word and bare tokens are sub-token continuations — to a text-only instruct LLM (`Qwen/Qwen3-4B-Instruct-2507`, the non-thinking 2507 variant) and ask it to pick out meaningful contiguous expressions and label each:

- `CVO` — concrete visual object with a physical location in the image
- `ICON` — named iconographic subject (mythological / religious / historical figure)
- `STYLE` — painterly or formal attribute (brushwork, palette, technique, period name)
- `SPATIAL` — compositional/relational descriptor (background, foreground, left side)
- `AFFECT` — affective or interpretive claim (dramatic atmosphere, serene)
- `META` — artist name, painting title, or provenance metadata

```bash
python -m tamart.experiments.classify
# or use the bf16 weights instead of the default FP8:
python -m tamart.experiments.classify --model-name Qwen/Qwen3-4B-Instruct-2507
```

Paintings are processed one at a time. The token indices in the output stay consistent with the `token_labels` in the corresponding describe answer, so each span maps straight onto its TAM activation maps. Output layout (mirrors `describe`, one subfolder per painting):

```
<data-dir>/classify/<image_stem>/
  classification.json  # JSON list of {tokens: [indices], word, category} spans
  raw_response.txt     # raw model output, kept for debugging parse failures
```

Spans with out-of-range indices or unknown categories are dropped; a caption whose response can't be parsed is logged as `[parse-failed]` and skipped. The script is resumable — `--no-overwrite` (the default) skips paintings that already have a `classification.json`.

## Layout

```
src/tamart/tam/
  core.py        — the TAM algorithm (rank Gaussian filter, ECI, latex vis)
  explainer.py   — TAMExplainer class (Qwen2-VL wrapper)
  eval.py        — IoU / ROUGE-L / METEOR metrics over TAM maps
  qwen_utils.py  — Qwen vision pre-processing helpers
src/tamart/method/
  compute.py     — precompute per-image TAM maps, save as safetensors
src/tamart/experiments/
  describe.py    — caption every painting, save TAM maps for notebook replay
  classify.py    — group description tokens into art-description categories
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
