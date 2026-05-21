"""Check the annotations produced by ``tamart.data.annotate``.

Two phases:

1. **Parse** each raw model response into a structured object.

   ``title_artist`` responses become ``{"title": <str>, "artist": <str>}`` and
   ``artist`` responses become ``{"artist": <str>}``. Responses that don't
   contain a usable JSON object are flagged with ``result: "parse_error"`` so
   the entry stays visible in the output but is excluded from judging.

2. **Judge** each parsed prediction against the ground-truth ``annotations.json``
   using a Qwen3-4B Instruct text-only LLM (the 2507 non-thinking variant by
   default — picked because the task is short structured classification).

   For ``title_artist`` the recorded ``result`` is one of:
   ``"both_correct" | "title_only" | "artist_only" | "both_wrong"``.
   For ``artist`` it is ``"correct" | "wrong"``.

Output: ``<DATA_DIR>/<SOURCE_MODEL>/results_clean.json``. The script is
resumable — entries that already carry a non-``parse_error`` result are
skipped on re-run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import click
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import tamart  # configures HF_HOME before transformers loads weights  # noqa: F401


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "datasets" / "wikiart_most_viewed"

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str | None) -> dict[str, Any] | None:
    """Pull a JSON object out of the model's raw answer.

    Accepts plain JSON, JSON wrapped in a ``` / ```json fence, or JSON with
    extra prose around it. Returns ``None`` on any parse failure.
    """
    if not raw:
        return None
    text = raw.strip()
    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        m = _BARE_OBJ_RE.search(text)
        candidate = m.group(0) if m else text
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _coerce_str(v: Any) -> str | None:
    if isinstance(v, str):
        s = v.strip()
        return s or None
    return None


def _parse_title_artist(raw: str | None) -> dict[str, str] | None:
    obj = _extract_json(raw)
    if obj is None:
        return None
    title = _coerce_str(obj.get("title"))
    artist = _coerce_str(obj.get("artist"))
    if not title or not artist:
        return None
    return {"title": title, "artist": artist}


def _parse_artist(raw: str | None) -> dict[str, str] | None:
    obj = _extract_json(raw)
    if obj is None:
        return None
    artist = _coerce_str(obj.get("artist"))
    if not artist:
        return None
    return {"artist": artist}


def _load_ground_truth(annotations_path: Path) -> dict[str, dict[str, str]]:
    """Index ``annotations.json`` by the image's basename (matches results.json keys)."""
    with open(annotations_path, encoding="utf-8") as f:
        anns = json.load(f)
    out: dict[str, dict[str, str]] = {}
    for a in anns:
        img_rel = a.get("imagePath")
        if not img_rel:
            continue
        basename = Path(img_rel).name
        out[basename] = {
            "title": a.get("title") or "",
            "artist": a.get("artistName") or "",
        }
    return out


_JUDGE_TITLE_ARTIST_TEMPLATE = (
    "You are grading whether a model correctly identified a famous painting.\n"
    "\n"
    "Ground truth:\n"
    "  title:  {gt_title!r}\n"
    "  artist: {gt_artist!r}\n"
    "\n"
    "Model prediction:\n"
    "  title:  {pred_title!r}\n"
    "  artist: {pred_artist!r}\n"
    "\n"
    "Decide independently whether the predicted title and the predicted artist "
    "refer to the same artwork / same person as the ground truth. Be lenient "
    "about:\n"
    "  - articles and punctuation (\"The Starry Night\" == \"Starry Night\")\n"
    "  - common translations and alternate titles\n"
    "  - name ordering (\"da Vinci Leonardo\" == \"Leonardo da Vinci\")\n"
    "  - diacritics and minor spelling differences\n"
    "Be strict about:\n"
    "  - different works by the same artist (titles must match the actual painting)\n"
    "  - different artists (a wrong attribution is wrong even if the school matches)\n"
    "\n"
    'Reply with a single JSON object and nothing else: '
    '{{"title_correct": true|false, "artist_correct": true|false}}.'
)


_JUDGE_ARTIST_TEMPLATE = (
    "You are grading whether a model correctly identified the artist of a famous painting.\n"
    "\n"
    "Ground truth artist: {gt_artist!r}\n"
    "Model prediction:    {pred_artist!r}\n"
    "\n"
    "Be lenient about name ordering (\"da Vinci Leonardo\" == \"Leonardo da Vinci\"), "
    "diacritics, and minor spelling differences. Be strict about wrong attributions.\n"
    "\n"
    'Reply with a single JSON object and nothing else: {{"artist_correct": true|false}}.'
)


@torch.inference_mode()
def _judge(model, tokenizer, prompt: str, max_new_tokens: int) -> dict[str, Any] | None:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _extract_json(raw)


def _classify_title_artist(judgment: dict[str, Any] | None) -> str:
    if not judgment:
        return "judge_error"
    t = bool(judgment.get("title_correct"))
    a = bool(judgment.get("artist_correct"))
    if t and a:
        return "both_correct"
    if t and not a:
        return "title_only"
    if a and not t:
        return "artist_only"
    return "both_wrong"


def _classify_artist(judgment: dict[str, Any] | None) -> str:
    if not judgment:
        return "judge_error"
    return "correct" if bool(judgment.get("artist_correct")) else "wrong"


def _save_state(out_path: Path, state: dict[str, Any]) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(out_path)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Parse the model's annotation responses and grade them against the "
        "WikiArt ground truth using a small Qwen3 LLM judge. Writes "
        "DATA_DIR/<source-model>/results_clean.json."
    ),
)
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=_DEFAULT_DATA_DIR,
    show_default=True,
    help="Dataset root containing annotations.json and the source-model folder.",
)
@click.option(
    "--source-model",
    default="Qwen/Qwen2-VL-2B-Instruct",
    show_default=True,
    help="Source VL model whose results.json we're cleaning (subfolder name).",
)
@click.option(
    "--judge-model",
    default="Qwen/Qwen3-4B-Instruct-2507",
    show_default=True,
    help="Text-only LLM used to compare predictions against ground truth.",
)
@click.option(
    "--max-new-tokens",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
    help="Generation cap per judging call.",
)
def main(
    data_dir: Path,
    source_model: str,
    judge_model: str,
    max_new_tokens: int,
) -> None:
    data_dir = data_dir.expanduser().resolve()
    annotations_path = data_dir / "annotations.json"
    source_dir = data_dir / source_model
    results_path = source_dir / "results.json"
    clean_path = source_dir / "results_clean.json"

    if not annotations_path.exists():
        raise click.ClickException(f"Missing ground truth: {annotations_path}")
    if not results_path.exists():
        raise click.ClickException(f"Missing model results: {results_path}")

    with open(results_path, encoding="utf-8") as f:
        raw_state = json.load(f)
    raw_results: dict[str, dict[str, Any]] = raw_state.get("results", {})
    gt = _load_ground_truth(annotations_path)

    # ---- Phase 1: parse -----------------------------------------------------
    clean_state: dict[str, Any] = {}
    if clean_path.exists():
        try:
            with open(clean_path, encoding="utf-8") as f:
                clean_state = json.load(f)
        except (json.JSONDecodeError, OSError):
            clean_state = {}
    clean_state["source_model"] = source_model
    clean_state["judge_model"] = judge_model
    clean_state.setdefault("results", {})
    clean_results: dict[str, dict[str, Any]] = clean_state["results"]

    n_ta_parsed = n_ta_total = 0
    n_a_parsed = n_a_total = 0

    for filename, entry in raw_results.items():
        out_entry = clean_results.get(filename, {})

        if "title_artist" in entry:
            n_ta_total += 1
            parsed = _parse_title_artist(entry.get("title_artist"))
            if parsed:
                n_ta_parsed += 1
                # Preserve existing judgment if we already graded this row.
                existing = out_entry.get("title_artist")
                if existing and existing.get("result") not in (None, "parse_error"):
                    out_entry["title_artist"] = {**parsed, "result": existing["result"]}
                else:
                    out_entry["title_artist"] = {**parsed}
            else:
                out_entry["title_artist"] = {"result": "parse_error"}

        if "artist" in entry:
            n_a_total += 1
            parsed = _parse_artist(entry.get("artist"))
            if parsed:
                n_a_parsed += 1
                existing = out_entry.get("artist")
                if existing and existing.get("result") not in (None, "parse_error"):
                    out_entry["artist"] = {**parsed, "result": existing["result"]}
                else:
                    out_entry["artist"] = {**parsed}
            else:
                out_entry["artist"] = {"result": "parse_error"}

        clean_results[filename] = out_entry

    click.echo(
        f"Parsed title_artist: {n_ta_parsed}/{n_ta_total}  |  "
        f"Parsed artist: {n_a_parsed}/{n_a_total}"
    )
    _save_state(clean_path, clean_state)

    # ---- Phase 2: judge -----------------------------------------------------
    # Collect rows that need judging (parsed + not yet judged + ground truth available).
    ta_pending: list[str] = []
    a_pending: list[str] = []
    for filename, entry in clean_results.items():
        if filename not in gt:
            continue
        ta = entry.get("title_artist")
        if ta and "title" in ta and "artist" in ta and "result" not in ta:
            ta_pending.append(filename)
        a = entry.get("artist")
        if a and "artist" in a and "result" not in a:
            a_pending.append(filename)

    click.echo(
        f"To judge — title_artist: {len(ta_pending)}, artist: {len(a_pending)}"
    )
    if not ta_pending and not a_pending:
        click.echo("Nothing to judge.")
        return

    click.echo(f"Loading judge: {judge_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    model = AutoModelForCausalLM.from_pretrained(
        judge_model, torch_dtype="auto", device_map="auto"
    )

    try:
        for filename in tqdm(ta_pending, desc="Judging title_artist", unit="row"):
            ta = clean_results[filename]["title_artist"]
            prompt = _JUDGE_TITLE_ARTIST_TEMPLATE.format(
                gt_title=gt[filename]["title"],
                gt_artist=gt[filename]["artist"],
                pred_title=ta["title"],
                pred_artist=ta["artist"],
            )
            judgment = _judge(model, tokenizer, prompt, max_new_tokens)
            ta["result"] = _classify_title_artist(judgment)
            _save_state(clean_path, clean_state)

        for filename in tqdm(a_pending, desc="Judging artist", unit="row"):
            a = clean_results[filename]["artist"]
            prompt = _JUDGE_ARTIST_TEMPLATE.format(
                gt_artist=gt[filename]["artist"],
                pred_artist=a["artist"],
            )
            judgment = _judge(model, tokenizer, prompt, max_new_tokens)
            a["result"] = _classify_artist(judgment)
            _save_state(clean_path, clean_state)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Summary -----------------------------------------------------------
    ta_counts: dict[str, int] = {}
    a_counts: dict[str, int] = {}
    for entry in clean_results.values():
        ta = entry.get("title_artist")
        if ta and "result" in ta:
            ta_counts[ta["result"]] = ta_counts.get(ta["result"], 0) + 1
        a = entry.get("artist")
        if a and "result" in a:
            a_counts[a["result"]] = a_counts.get(a["result"], 0) + 1

    click.echo(f"title_artist results: {ta_counts}")
    click.echo(f"artist results:       {a_counts}")
    click.echo(f"Wrote: {clean_path}")


if __name__ == "__main__":
    main()
