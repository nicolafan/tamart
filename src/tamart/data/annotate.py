"""Annotate WikiArt paintings with Qwen2-VL responses to two probe prompts.

Loads the same multimodal model the TAM explainer uses and asks two prompts
per image:

  1. Title + artist — answer template: "The artwork is named [title] by [artist]."
  2. Artist only    — answer template: "The artist who made this artwork is [artist]."

Responses are written incrementally to ``<DATA_DIR>/<MODEL_NAME>/results.json``
(the model name's ``/`` becomes a nested directory). Re-running the script
skips images whose entry already has both prompt answers, so an interrupted
run resumes from the first unfinished image.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

import tamart  # configures HF_HOME before transformers loads weights  # noqa: F401
from tamart.tam.qwen_utils import process_vision_info


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "datasets" / "wikiart_most_viewed"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PROMPTS: dict[str, str] = {
    "title_artist": (
        "Identify this artwork. Respond with a single JSON object and nothing "
        'else, in this exact shape: {"title": "<the artwork title>", '
        '"artist": "<the artist full name>"}.'
    ),
    "artist": (
        "Who made this artwork? Respond with a single JSON object and nothing "
        'else, in this exact shape: {"artist": "<the artist full name>"}.'
    ),
}


def _load_model(model_name: str):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_name, min_pixels=min_pixels, max_pixels=max_pixels
    )
    return model, processor


@torch.inference_mode()
def _generate(model, processor, image_path: Path, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, padding=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, use_cache=True
    )
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, out)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def _load_existing(out_path: Path) -> dict[str, Any]:
    if not out_path.exists():
        return {}
    try:
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(out_path: Path, state: dict[str, Any]) -> None:
    """Atomic write so an interrupt mid-dump can't corrupt the JSON."""
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(out_path)


def _entry_is_complete(entry: dict[str, Any] | None) -> bool:
    if not entry:
        return False
    return all(entry.get(k) for k in PROMPTS)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Annotate each painting in DATA_DIR/images with Qwen2-VL answers to "
        "two probe prompts. Results are written to "
        "DATA_DIR/<model-name>/results.json and the script resumes from "
        "where it left off when re-run."
    ),
)
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=_DEFAULT_DATA_DIR,
    show_default=True,
    help="Dataset root containing images/ and (after a run) <model>/results.json.",
)
@click.option(
    "--model-name",
    default="Qwen/Qwen2-VL-2B-Instruct",
    show_default=True,
    help="HF model id; the literal string becomes a nested subdir under data-dir.",
)
@click.option(
    "--images-subdir",
    default="images",
    show_default=True,
    help="Subdirectory of DATA_DIR holding the input images.",
)
@click.option(
    "--max-new-tokens",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
    help="Generation cap per prompt.",
)
def main(
    data_dir: Path,
    model_name: str,
    images_subdir: str,
    max_new_tokens: int,
) -> None:
    data_dir = data_dir.expanduser().resolve()
    images_dir = data_dir / images_subdir
    if not images_dir.is_dir():
        raise click.ClickException(f"Images directory not found: {images_dir}")

    out_dir = data_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    state = _load_existing(out_path)
    state["model"] = model_name
    state["prompts"] = PROMPTS
    state.setdefault("results", {})
    results: dict[str, dict[str, Any]] = state["results"]

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
    )
    pending = [p for p in image_paths if not _entry_is_complete(results.get(p.name))]

    click.echo(
        f"Total images: {len(image_paths)};  already done: "
        f"{len(image_paths) - len(pending)};  pending: {len(pending)}"
    )
    if not pending:
        click.echo("Nothing to do.")
        return

    click.echo(f"Loading {model_name} ...")
    model, processor = _load_model(model_name)

    try:
        for img_path in tqdm(pending, desc="Annotating", unit="image"):
            entry = results.get(img_path.name, {})
            for key, prompt in PROMPTS.items():
                if entry.get(key):
                    continue
                try:
                    entry[key] = _generate(
                        model, processor, img_path, prompt, max_new_tokens
                    )
                except Exception as e:  # noqa: BLE001
                    tqdm.write(f"[warn] {img_path.name} / {key}: {e}")
                    entry.setdefault("_errors", {})[key] = repr(e)
            results[img_path.name] = entry
            _save_state(out_path, state)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    click.echo(f"Done. Results: {out_path}")


if __name__ == "__main__":
    main()
