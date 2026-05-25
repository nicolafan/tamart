"""Experiment: open-vocabulary segmentation of CVO/ICON spans with SAM 3.

The ``classify`` phase (:mod:`tamart.experiments.classify`) tagged every span
of each painting's description with a category. Here we take the spans labelled
**CVO** (concrete visual object) and **ICON** (named iconographic subject) and,
using their surface ``word`` as a *concept prompt*, segment the painting with
Meta AI's **SAM 3** (``facebook/sam3``). SAM 3 performs Promptable Concept
Segmentation: given a short noun phrase it returns instance masks for *every*
matching object in the image. We collapse those instances into a single binary
mask per expression (the union of all instances above the score threshold) — a
pseudo-ground-truth localisation for "where in the painting is this concept".

SAM 3 is run on the **processed image** that TAM saw
(``describe/<stem>/proc_img.png``), so the saved masks live on exactly the same
canvas as the per-token TAM activation maps and can be compared pixel-for-pixel
without any aspect-ratio juggling.

Output layout (mirrors ``describe`` / ``classify``, one subfolder per painting)::

    data/datasets/wikiart_most_viewed/segment/<stem>/
      masks.safetensors   # one uint8 {0,1} union mask per span, key = span index
      segments.json       # per-span metadata: tokens, word, category, scores, ...

``masks.safetensors`` keys are the span's index in ``classification.json`` (as a
string), so a notebook can line a mask up with its span — and therefore with
the span's TAM maps — directly. Spans that produced no detections still get an
all-zero mask, so every CVO/ICON span has an entry.

The script is resumable — ``--no-overwrite`` (the default) skips paintings that
already have a ``segments.json``.

.. note::
   SAM 3 requires ``transformers>=5`` (its config is saved as ``5.0.0.dev0``),
   which conflicts with the repo's pinned ``transformers<5`` (vllm 0.16.0). Run
   this script from the dedicated ``.venv-sam3`` environment, e.g.::

       .venv-sam3/bin/python -m tamart.experiments.segment --limit 2
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file as st_save_file
from tqdm import tqdm

# Importing tamart first pins HF_HOME to <repo>/data/hf (see tamart/__init__).
import tamart  # noqa: F401  (side effect: configures HF_HOME)
from transformers import Sam3Model, Sam3Processor

# Categories we segment. STYLE / AFFECT / META describe how the painting looks
# or what it is called, not a thing with a physical location, so they have no
# meaningful spatial ground truth and are skipped.
_SEGMENT_CATEGORIES = ("CVO", "ICON")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "datasets" / "wikiart_most_viewed"

_MODEL_ID = "facebook/sam3"


def _list_classify_dirs(classify_root: Path) -> list[Path]:
    return sorted(
        p for p in classify_root.iterdir()
        if p.is_dir() and (p / "classification.json").exists()
    )


def _segment_spans(classification: list[dict]) -> list[tuple[int, dict]]:
    """Return ``(span_index, span)`` for the CVO/ICON spans, keeping the span's
    original index in ``classification.json`` as the stable key."""
    return [
        (i, s) for i, s in enumerate(classification)
        if s.get("category") in _SEGMENT_CATEGORIES
        and isinstance(s.get("word"), str) and s["word"].strip()
    ]


class ConceptSegmenter:
    """SAM 3 wrapped for repeated single-image, multi-prompt concept queries."""

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        device: str | None = None,
        dtype: str = "bfloat16",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if self.device == "cuda" else torch.float32
        self.model = Sam3Model.from_pretrained(model_id, dtype=torch_dtype).to(
            self.device
        )
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_id)

    @torch.no_grad()
    def segment_image(
        self,
        image: Image.Image,
        prompts: list[str],
        score_threshold: float,
        mask_threshold: float,
    ) -> list[dict]:
        """Segment ``image`` once per text prompt.

        The image's vision features are computed a single time and reused across
        every prompt (SAM 3's recommended multi-prompt-on-one-image pattern).

        Returns, per prompt, a dict with the union mask (``np.uint8`` ``[H, W]``
        in {0, 1}), the per-instance ``scores`` and ``boxes``, and the mask
        height/width.
        """
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        target_sizes = img_inputs["original_sizes"].tolist()  # [[H, W]]
        h, w = int(target_sizes[0][0]), int(target_sizes[0][1])
        vision_embeds = self.model.get_vision_features(
            pixel_values=img_inputs["pixel_values"]
        )

        # SAM 3's text encoder has only 32 position embeddings, and
        # ``Sam3Processor`` pads short prompts to 32 but does NOT truncate long
        # ones (it ignores ``truncation`` kwargs), so an over-long classify
        # "word" (e.g. a whole transcribed clause) overflows the encoder and
        # crashes. Tokenise directly with explicit truncation — for prompts that
        # fit, this is byte-identical to the processor's text output.
        tok = self.processor.tokenizer
        results = []
        for prompt in prompts:
            text_inputs = tok(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
            ).to(self.device)
            outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
            post = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=score_threshold,
                mask_threshold=mask_threshold,
                target_sizes=target_sizes,
            )[0]

            masks = post["masks"]      # bool/float tensor [N, H, W] (orig size)
            scores = post["scores"]    # [N]
            boxes = post["boxes"]      # [N, 4] xyxy abs pixels

            if masks is not None and len(masks) > 0:
                m = masks.to(torch.bool).any(dim=0).cpu().numpy().astype(np.uint8)
                scores_list = [float(s) for s in scores.tolist()]
                boxes_list = [[float(c) for c in b] for b in boxes.tolist()]
            else:
                m = np.zeros((h, w), dtype=np.uint8)
                scores_list, boxes_list = [], []

            results.append(
                {
                    "mask": m,
                    "scores": scores_list,
                    "boxes": boxes_list,
                    "height": h,
                    "width": w,
                }
            )
        return results


@click.command(context_settings={"show_default": True})
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_DATA_DIR,
    help="Dataset root containing the classify/ and describe/ subfolders.",
)
@click.option(
    "--classify-name", default="classify",
    help="Subfolder with the classify-phase spans to read.",
)
@click.option(
    "--describe-name", default="describe",
    help="Subfolder with the describe-phase proc_img.png to segment.",
)
@click.option(
    "--out-name", default="segment",
    help="Subfolder inside the dataset root to write masks into.",
)
@click.option(
    "--model-id", default=_MODEL_ID,
    help="Hugging Face model id for the SAM 3 concept segmenter.",
)
@click.option(
    "--score-threshold", type=float, default=0.5,
    help="Minimum SAM 3 instance confidence to keep a detection.",
)
@click.option(
    "--mask-threshold", type=float, default=0.5,
    help="Probability cutoff for binarising each instance mask.",
)
@click.option(
    "--limit", type=int, default=None,
    help="Only process the first N paintings (smoke testing).",
)
@click.option(
    "--overwrite/--no-overwrite", default=False,
    help="If False, skip paintings that already have a segments.json.",
)
def main(
    data_dir: Path,
    classify_name: str,
    describe_name: str,
    out_name: str,
    model_id: str,
    score_threshold: float,
    mask_threshold: float,
    limit: int | None,
    overwrite: bool,
):
    classify_root = data_dir / classify_name
    describe_root = data_dir / describe_name
    out_root = data_dir / out_name
    out_root.mkdir(parents=True, exist_ok=True)

    dirs = _list_classify_dirs(classify_root)
    if not overwrite:
        dirs = [
            d for d in dirs
            if not (out_root / d.name / "segments.json").exists()
        ]
    if limit is not None:
        dirs = dirs[:limit]

    if not dirs:
        click.echo("Nothing to do — all paintings already segmented.")
        return

    click.echo(f"Segmenting {len(dirs)} paintings with SAM 3 ({model_id})")
    segmenter = ConceptSegmenter(model_id=model_id)

    for d in tqdm(dirs):
        proc_img_path = describe_root / d.name / "proc_img.png"
        if not proc_img_path.exists():
            tqdm.write(f"[skip] {d.name}: no proc_img.png")
            continue

        classification = json.loads((d / "classification.json").read_text())
        spans = _segment_spans(classification)

        out_dir = out_root / d.name
        out_dir.mkdir(parents=True, exist_ok=True)

        if not spans:
            (out_dir / "segments.json").write_text(json.dumps([], indent=2))
            continue

        image = Image.open(proc_img_path).convert("RGB")
        results = segmenter.segment_image(
            image,
            [s["word"] for _, s in spans],
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )

        tensor_dict, meta = {}, []
        for (span_idx, span), res in zip(spans, results):
            tensor_dict[str(span_idx)] = torch.from_numpy(res["mask"]).contiguous()
            meta.append(
                {
                    "span_index": span_idx,
                    "tokens": span["tokens"],
                    "word": span["word"],
                    "category": span["category"],
                    "n_instances": len(res["scores"]),
                    "scores": res["scores"],
                    "boxes": res["boxes"],
                    "height": res["height"],
                    "width": res["width"],
                    "mask_area": int(res["mask"].sum()),
                }
            )

        st_save_file(tensor_dict, str(out_dir / "masks.safetensors"))
        (out_dir / "segments.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
