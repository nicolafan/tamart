"""Experiment: describe each WikiArt image with Qwen2-VL + TAM.

For every image in ``data/datasets/wikiart_most_viewed/images`` we ask the
explaining MLLM to write a detailed description of the painting. The
generated text, per-token activation maps, and the processed image used by
TAM are written into ``data/datasets/wikiart_most_viewed/describe/<stem>/``
so that :meth:`TAMExplainer.explain_interactive_precomputed` can replay the
annotation in a notebook without re-running inference.

Supports batched generation via ``--batch-size``.
"""

from __future__ import annotations

from pathlib import Path

import click
from tqdm import tqdm

from tamart.tam import TAMExplainer

DESCRIBE_PROMPT = "Describe the content and style of this image."

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "datasets" / "wikiart_most_viewed"


def _list_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)


@click.command(context_settings={"show_default": True})
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_DATA_DIR,
    help="Dataset root containing an 'images/' subfolder.",
)
@click.option(
    "--out-name",
    default="describe",
    help="Subfolder inside the dataset root to write annotations into.",
)
@click.option("--batch-size", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=256)
@click.option(
    "--model-name", default="Qwen/Qwen2-VL-2B-Instruct",
    help="Hugging Face model id for the explaining MLLM.",
)
@click.option(
    "--overwrite/--no-overwrite", default=False,
    help="If False, skip images that already have an answer.json.",
)
def main(
    data_dir: Path,
    out_name: str,
    batch_size: int,
    max_new_tokens: int,
    model_name: str,
    overwrite: bool,
):
    images_dir = data_dir / "images"
    out_root = data_dir / out_name
    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = _list_images(images_dir)
    if not overwrite:
        image_paths = [
            p for p in image_paths
            if not (out_root / p.stem / "answer.json").exists()
        ]

    if not image_paths:
        click.echo("Nothing to do — all images already annotated.")
        return

    click.echo(
        f"Annotating {len(image_paths)} images with batch_size={batch_size}"
    )

    explainer = TAMExplainer(model_name=model_name)

    pbar = tqdm(total=len(image_paths))
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        results = explainer.explain_batch(
            [str(p) for p in batch_paths],
            DESCRIBE_PROMPT,
            max_new_tokens=max_new_tokens,
        )
        for path, res in zip(batch_paths, results):
            TAMExplainer.save_precomputed(out_root / path.stem, res)
        pbar.update(len(batch_paths))
    pbar.close()


if __name__ == "__main__":
    main()
