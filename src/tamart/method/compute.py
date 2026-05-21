"""Compute per-image TAM maps and save them as safetensors.

For every image in ``<data-dir>/<images-subdir>``:

  1. Run Qwen2-VL with the artist-only JSON prompt.
  2. Identify the generated tokens whose decoded char-span overlaps the parsed
     artist name.
  3. Merge those tokens' per-pixel max activation into a single saliency map,
     normalized to ``[0, 1]`` float32 at the model's native patch grid.
  4. Save the map to
     ``<data-dir>/<model-name>/maps/<image_stem>.safetensors``.

``--token-mode all`` (default) merges every artist-name token's map; ``first``
keeps only the first such token. Images that already have a maps file are
skipped, so re-runs resume from the first unfinished image. Images whose
answer doesn't contain a parseable artist name or whose artist name doesn't
align to any generated tokens are skipped with a warning.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import torch
from safetensors.numpy import save_file
from tqdm import tqdm

import tamart  # configures HF_HOME before transformers loads weights  # noqa: F401
from tamart.data.check import _parse_artist
from tamart.experiments.describe import (
    ARTIST_PROMPT,
    TOKEN_MODES,
    _find_artist_token_indices,
)
from tamart.tam import TAMExplainer


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "datasets" / "wikiart_most_viewed"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _merge_map_unit(maps: list[np.ndarray], indices: list[int]) -> np.ndarray:
    """Per-pixel max of selected token maps, returned as float32 in [0, 1].

    ``TAMExplainer`` returns maps as uint8 in [0, 255] (the rank-Gaussian-
    filtered, min-max-normalized output of ``multimodal_process``). Dividing
    by 255 recovers the native [0, 1] scale.
    """
    selected = [maps[i] for i in indices]
    merged = np.maximum.reduce(selected).astype(np.float32) / 255.0
    return np.clip(merged, 0.0, 1.0)


def _save_map(
    out_path: Path,
    merged_map: np.ndarray,
    *,
    model_name: str,
    prompt: str,
    generated_text: str,
    artist: str,
    artist_token_indices: list[int],
    selected_token_indices: list[int],
    token_mode: str,
) -> None:
    """Atomic write via ``.part`` swap; safetensors metadata is str-keyed."""
    metadata: dict[str, str] = {
        "model": model_name,
        "prompt": prompt,
        "text": generated_text,
        "artist": artist,
        "token_mode": token_mode,
        "artist_token_indices": json.dumps(artist_token_indices),
        "selected_token_indices": json.dumps(selected_token_indices),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    save_file({"map": merged_map}, str(tmp), metadata=metadata)
    tmp.replace(out_path)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Compute and save a TAM saliency map per image as safetensors. By "
        "default the map merges every artist-name token (per-pixel max), "
        "normalized to [0, 1] float32 at the model's native patch grid. "
        "Results land under <data-dir>/<model-name>/maps/."
    ),
)
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=_DEFAULT_DATA_DIR,
    show_default=True,
    help="Dataset root containing the images subdir.",
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
    "--prompt",
    default=ARTIST_PROMPT,
    show_default=False,
    help="Prompt to send with each image. Defaults to the exp1 artist-JSON prompt.",
)
@click.option(
    "--token-mode",
    type=click.Choice(TOKEN_MODES),
    default="all",
    show_default=True,
    help=(
        "Which artist-name token(s) drive the saved map: 'all' merges every "
        "token overlapping the artist name (per-pixel max), 'first' uses "
        "only the first such token."
    ),
)
@click.option(
    "--max-new-tokens",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
    help="Generation cap.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Process at most N pending images (handy for smoke tests).",
)
def main(
    data_dir: Path,
    model_name: str,
    images_subdir: str,
    prompt: str,
    token_mode: str,
    max_new_tokens: int,
    limit: int | None,
) -> None:
    data_dir = data_dir.expanduser().resolve()
    images_dir = data_dir / images_subdir
    if not images_dir.is_dir():
        raise click.ClickException(f"Images directory not found: {images_dir}")

    maps_dir = data_dir / model_name / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
    )
    pending = [
        p for p in image_paths
        if not (maps_dir / f"{p.stem}.safetensors").exists()
    ]
    if limit is not None:
        pending = pending[:limit]

    click.echo(
        f"Total images: {len(image_paths)};  "
        f"done: {len(image_paths) - len(pending)};  pending: {len(pending)}"
    )
    if not pending:
        click.echo("Nothing to do.")
        return

    click.echo(f"Loading {model_name} ...")
    explainer = TAMExplainer(model_name=model_name)

    try:
        for img_path in tqdm(pending, desc="Computing maps", unit="image"):
            out_path = maps_dir / f"{img_path.stem}.safetensors"
            try:
                _process_image(
                    explainer,
                    img_path,
                    out_path,
                    prompt=prompt,
                    token_mode=token_mode,
                    max_new_tokens=max_new_tokens,
                    model_name=model_name,
                )
            except Exception as e:  # noqa: BLE001
                tqdm.write(f"[warn] {img_path.name}: {e!r}")
    finally:
        del explainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    click.echo(f"Done. Maps under: {maps_dir}")


def _process_image(
    explainer: TAMExplainer,
    img_path: Path,
    out_path: Path,
    *,
    prompt: str,
    token_mode: str,
    max_new_tokens: int,
    model_name: str,
) -> None:
    result = explainer.explain(
        image=str(img_path), prompt=prompt, max_new_tokens=max_new_tokens
    )
    parsed = _parse_artist(result["text"])
    if not parsed:
        tqdm.write(f"[skip] {img_path.name}: unparseable answer {result['text']!r}")
        return
    artist = parsed["artist"]

    tokens = result["tokens"]
    token_ids = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
    indices = _find_artist_token_indices(explainer, token_ids, artist)
    if not indices:
        tqdm.write(
            f"[skip] {img_path.name}: artist {artist!r} not aligned to any token"
        )
        return
    selected = [indices[0]] if token_mode == "first" else indices

    merged = _merge_map_unit(result["maps"], selected)
    _save_map(
        out_path,
        merged,
        model_name=model_name,
        prompt=prompt,
        generated_text=result["text"],
        artist=artist,
        artist_token_indices=indices,
        selected_token_indices=selected,
        token_mode=token_mode,
    )


if __name__ == "__main__":
    main()
