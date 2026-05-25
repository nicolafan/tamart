"""Build static assets for the TAMArt project page (GitHub Pages).

Reads the precomputed pipeline outputs under
``data/datasets/wikiart_most_viewed/{describe,classify,segment,validate_meta}``
and emits a self-contained ``assets/`` tree the static site loads lazily:

  assets/data/index.json                 -> gallery list + per-painting summary
  assets/data/paintings/<stem>.json      -> caption segments, per-span TAM maps
  assets/img/<stem>.jpg                  -> resized painting
  assets/masks/<stem>__<span>.png        -> SAM 3 mask (CVO/ICON, resized)

Run from the repo root with the project venv (needs torch/safetensors/PIL/numpy):
    .venv/bin/python scripts/build_site_assets.py --n 150
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file

import tamart  # noqa: F401  (sets HF_HOME before transformers import)
from transformers import AutoProcessor

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data" / "datasets" / "wikiart_most_viewed"
DESCRIBE = DATASET / "describe"
CLASSIFY = DATASET / "classify"
SEGMENT = DATASET / "segment"
VALIDATE = DATASET / "validate_meta"
IMAGES = DATASET / "images"
ANNOTATIONS = DATASET / "annotations.json"

OUT = REPO / "_site" / "assets"
MAX_DIM = 1000           # longest side of the displayed painting (px)
MASK_MAX_DIM = 1000      # SAM masks rendered to match
JPEG_Q = 85

EOS_IDS = {151645, 151643}  # <|im_end|>, <|endoftext|>
SEG_CATS = {"CVO", "ICON"}

# Canonical works we always want present (matched by stem substring).
CANONICAL = [
    "mona-lisa", "the-starry-night", "in-bed-the-kiss", "the-birth-of-venus",
    "the-two-fridas", "the-nightwatch", "the-scream", "girl-with-a-pearl",
    "guernica", "the-kiss", "las-meninas", "the-last-supper",
    "the-luncheon-on-the-grass", "effect-of-snow-at-petit-montrouge",
    "the-school-of-athens", "american-gothic", "the-garden-of-earthly",
    "sistine-madonna", "the-creation-of-adam", "impression-sunrise",
]


def load_caption_pieces(tokenizer, tokens):
    """Trim padding/EOS, return (generated_ids, per-token decoded text pieces)."""
    gen = []
    for t in tokens:
        if t in EOS_IDS:
            break
        gen.append(int(t))
    pieces = [tokenizer.decode([t]) for t in gen]
    return gen, pieces


def build_segments(pieces, span_of):
    """Turn token pieces + a {token_index: span_id} map into caption segments.

    Each segment is {"t": text, "s": span_id_or_-1}. Consecutive tokens sharing
    a span (or sharing no span) are merged. Leading whitespace of a span is
    pushed into the preceding plain segment so highlight boxes start on a word.
    """
    segs = []
    for i, piece in enumerate(pieces):
        sid = span_of.get(i, -1)
        if sid != -1 and (i == 0 or span_of.get(i - 1, -1) != sid):
            # first token of a span: strip leading whitespace out of the box
            stripped = piece.lstrip()
            lead = piece[: len(piece) - len(stripped)]
            if lead:
                _push(segs, lead, -1)
            piece = stripped
        _push(segs, piece, sid)
    return segs


def _push(segs, text, sid):
    if not text:
        return
    if segs and segs[-1]["s"] == sid:
        segs[-1]["t"] += text
    else:
        segs.append({"t": text, "s": sid})


def span_map_uint8(maps, tokens):
    """Mean TAM map over a span's tokens, min-max normalized to uint8 [0,255]."""
    m = np.stack([maps[str(t)] for t in tokens], axis=0).mean(axis=0).astype(np.float32)
    lo, hi = float(m.min()), float(m.max())
    m = (m - lo) / (hi - lo + 1e-9)
    return np.clip(m * 255.0, 0, 255).astype(np.uint8)


def save_resized_image(src_img: Image.Image, dst: Path, max_dim: int):
    w, h = src_img.size
    scale = min(1.0, max_dim / max(w, h))
    out_w, out_h = round(w * scale), round(h * scale)
    img = src_img.convert("RGB").resize((out_w, out_h), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=JPEG_Q, optimize=True)
    return out_w, out_h


def save_mask_png(mask: np.ndarray, dst: Path, out_wh, color=(46, 204, 113), alpha=130):
    """Resize a {0,1} mask to (w,h) and write an RGBA PNG (tinted fg, clear bg)."""
    out_w, out_h = out_wh
    m = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(
        (out_w, out_h), Image.NEAREST
    )
    arr = np.asarray(m) > 127
    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    rgba[arr] = (*color, alpha)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(dst, optimize=True)


def stem_to_imagepath(stem):
    hits = list(IMAGES.glob(f"{stem}.*"))
    return hits[0] if hits else None


def select_stems(n):
    """Rank paintings by demo richness, keep canonical, fill for diversity."""
    annotations = json.loads(ANNOTATIONS.read_text())
    all_describe = sorted(p.name for p in DESCRIBE.iterdir() if (p / "answer.json").exists())

    # per-painting stats
    stats = {}
    for stem in all_describe:
        cls_path = CLASSIFY / stem / "classification.json"
        if not cls_path.exists():
            continue
        spans = json.loads(cls_path.read_text())
        cats = [s.get("category") for s in spans]
        seg_path = SEGMENT / stem / "segments.json"
        n_sam = 0
        if seg_path.exists():
            seg = json.loads(seg_path.read_text())
            n_sam = sum(1 for s in seg if s.get("n_instances", 0) > 0)
        stats[stem] = {
            "n_spans": len(spans),
            "n_icon": cats.count("ICON"),
            "n_sam": n_sam,
            "rank": int(stem.split("_")[0]),
        }

    def score(stem):
        s = stats[stem]
        return s["n_sam"] * 2 + s["n_icon"] * 4 + s["n_spans"] * 0.1

    chosen = []
    seen = set()

    def add(stem):
        if stem in stats and stem not in seen:
            seen.add(stem)
            chosen.append(stem)

    # 1) canonical works
    for key in CANONICAL:
        for stem in all_describe:
            if key in stem:
                add(stem)
                break

    # 2) richest demos: icon spans backed by a SAM detection, then most SAM masks
    rich = sorted(
        (s for s in stats if stats[s]["n_icon"] > 0 and stats[s]["n_sam"] > 0),
        key=score, reverse=True,
    )
    for stem in rich:
        if len(chosen) >= n:
            break
        add(stem)

    by_score = sorted(stats, key=score, reverse=True)
    for stem in by_score:
        if len(chosen) >= n:
            break
        add(stem)

    # 3) even stride across rank for period/genre diversity (replace tail if room)
    if len(chosen) < n:
        stride = max(1, len(all_describe) // (n - len(chosen) + 1))
        for stem in all_describe[::stride]:
            if len(chosen) >= n:
                break
            add(stem)

    chosen = chosen[:n]
    chosen.sort(key=lambda s: stats[s]["rank"])
    return chosen, annotations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.clean and OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data" / "paintings").mkdir(parents=True, exist_ok=True)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer ...")
    tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct").tokenizer

    stems, annotations = select_stems(args.n)
    rank_to_anno = {a["rank"]: a for a in annotations}
    print(f"Selected {len(stems)} paintings.")

    index = []
    cat_totals = {}
    for k, stem in enumerate(stems):
        rank = int(stem.split("_")[0])
        anno = rank_to_anno.get(rank, {})
        ans = json.loads((DESCRIBE / stem / "answer.json").read_text())
        spans = json.loads((CLASSIFY / stem / "classification.json").read_text())
        maps = load_file(DESCRIBE / stem / "maps.safetensors")

        img_src = stem_to_imagepath(stem)
        if img_src is None:
            print(f"  [skip] no image for {stem}")
            continue
        out_w, out_h = save_resized_image(Image.open(img_src), OUT / "img" / f"{stem}.jpg", MAX_DIM)

        # SAM masks keyed by span_index
        seg_meta = {}
        masks = {}
        seg_path = SEGMENT / stem / "segments.json"
        masks_path = SEGMENT / stem / "masks.safetensors"
        if seg_path.exists() and masks_path.exists():
            seg_meta = {s["span_index"]: s for s in json.loads(seg_path.read_text())}
            masks = load_file(masks_path)

        gen, pieces = load_caption_pieces(tokenizer, ans["tokens"])
        n_gen = len(gen)

        # Keep only spans with at least one in-range token, and remap their ids
        # so out_spans, the caption segments, and SAM masks all share one index.
        valid = []
        for sid, sp in enumerate(spans):
            toks = [t for t in sp["tokens"] if t < n_gen]
            if toks:
                valid.append((sid, sp, toks))
        span_of = {}  # token index -> new (compact) span id

        out_spans = []
        grid_h, grid_w = maps["0"].shape
        for new_sid, (sid, sp, toks) in enumerate(valid):
            for ti in toks:
                span_of[ti] = new_sid
            cat = sp.get("category", "?")
            cat_totals[cat] = cat_totals.get(cat, 0) + 1
            m = span_map_uint8(maps, toks)
            entry = {
                "word": sp["word"],
                "cat": cat,
                "map": base64.b64encode(m.flatten().tobytes()).decode("ascii"),
            }
            # SAM mask for this span (CVO/ICON with a detection); segments.json
            # is keyed by the ORIGINAL classification span index (sid).
            if cat in SEG_CATS and sid in seg_meta and seg_meta[sid].get("n_instances", 0) > 0:
                mk = str(sid)
                if mk in masks:
                    mfn = f"{stem}__{new_sid}.png"
                    save_mask_png(masks[mk], OUT / "masks" / mfn, (out_w, out_h))
                    entry["sam"] = mfn
                    entry["sam_n"] = int(seg_meta[sid]["n_instances"])
            out_spans.append(entry)

        segs = build_segments(pieces, span_of)

        vpath = VALIDATE / stem / "validation.json"
        meta = json.loads(vpath.read_text()) if vpath.exists() else {}

        painting = {
            "stem": stem,
            "image": f"assets/img/{stem}.jpg",
            "w": out_w, "h": out_h,
            "grid_h": int(grid_h), "grid_w": int(grid_w),
            "caption": ans["text"],
            "segments": segs,
            "spans": out_spans,
            "gt_title": anno.get("title"),
            "gt_artist": anno.get("artistName"),
            "year": anno.get("completitionYear"),
            "style": (anno.get("styles") or [None])[0],
            "genre": (anno.get("genres") or [None])[0],
            "meta": {
                "title_pred": meta.get("title_prediction"),
                "artist_pred": meta.get("artist_prediction"),
                "title_correct": meta.get("title_correct"),
                "artist_correct": meta.get("artist_correct"),
            },
        }
        (OUT / "data" / "paintings" / f"{stem}.json").write_text(json.dumps(painting))

        index.append({
            "stem": stem,
            "thumb": f"assets/img/{stem}.jpg",
            "title": anno.get("title") or stem,
            "artist": anno.get("artistName"),
            "year": anno.get("completitionYear"),
            "style": (anno.get("styles") or [None])[0],
            "n_spans": len(out_spans),
            "n_sam": sum(1 for s in out_spans if "sam" in s),
            "cats": sorted({s["cat"] for s in out_spans}),
        })
        if (k + 1) % 25 == 0:
            print(f"  ... {k + 1}/{len(stems)}")

    (OUT / "data" / "index.json").write_text(json.dumps({"paintings": index}))
    print(f"Done. {len(index)} paintings written.")
    print("Span category totals:", cat_totals)


if __name__ == "__main__":
    main()
