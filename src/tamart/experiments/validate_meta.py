"""Experiment: extract title/artist predictions from META spans and grade them.

The ``classify`` phase (:mod:`tamart.experiments.classify`) tagged every span
of each painting's description with a category. The spans labelled ``META`` are
the model's stabs at metadata — a title, an artist name, a date, etc. This
experiment turns those raw META spans into *graded* title/artist predictions,
reusing the same vLLM-served instruct LLM as ``classify``
(``Qwen/Qwen3-4B-Instruct-2507`` by default).

It runs in two stages, both driven by the same loaded model:

1. **Extract.** For each painting we collect its META span words and the full
   description text (from the ``describe`` answer), and ask the LLM which single
   META expression (if any) is the painting's TITLE and which is the ARTIST
   name. This yields at most one title and one artist prediction per painting.

2. **Grade.** For paintings that produced a prediction, we ask the LLM to
   compare the predicted title/artist against the ground truth in
   ``annotations.json`` — judging by meaning (case, punctuation, word order,
   reversed "Last First" name order all ignored). The verdict is written back
   onto the prediction.

The result for every painting is written to
``data/datasets/wikiart_most_viewed/validate_meta/<stem>/validation.json``,
mirroring the per-painting subfolder layout of ``classify`` / ``describe``.
A painting's ground-truth row is found by its rank, parsed from the ``NNNN_``
prefix of its folder name (robust to URL-slug normalisation differences).

vLLM batches each chunk concurrently (continuous batching); ``--batch-size`` is
the chunk size, which also sets how often results are checkpointed to disk for
resumability.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from tqdm import tqdm
from vllm import LLM, SamplingParams

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "datasets" / "wikiart_most_viewed"


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def _build_extract_system_prompt() -> str:
    return (
        "You are an expert at reading art metadata. You are given a painting's "
        "full description and a list of candidate metadata expressions (the "
        "'META' annotations) that were extracted from that description. A META "
        "expression is usually a painting title, an artist name, or a date.\n\n"
        "YOUR TASK: decide which single candidate (if any) is the painting's "
        "TITLE and which single candidate (if any) is the ARTIST's name.\n\n"
        "RULES:\n"
        "1. The title and artist MUST be copied verbatim from the candidate "
        "list. Never invent text and never pull a phrase from the description "
        "that is not in the candidate list.\n"
        "2. Use the description only as context to tell which candidate plays "
        "which role.\n"
        "3. A candidate that is only a date, location, gallery, or other "
        "provenance is neither a title nor an artist — leave that field null.\n"
        "4. Pick AT MOST ONE candidate per field. If no candidate is a title "
        "set \"title\" to null; if none is an artist set \"artist\" to null.\n\n"
        "Respond with ONLY a JSON object, no prose, no markdown fences:\n"
        '{"title": <string or null>, "artist": <string or null>}'
    )


_EXTRACT_USER_TEMPLATE = (
    "Description:\n{description}\n\n"
    "Candidate META expressions:\n{candidates}\n\n"
    "Which candidate is the title and which is the artist?"
)


def _build_grade_system_prompt() -> str:
    return (
        "You are an expert art historian checking predictions against ground "
        "truth. You are given a predicted painting TITLE and the true title, "
        "and a predicted ARTIST name and the true artist name. For each, decide "
        "whether the prediction refers to the SAME title / SAME artist as the "
        "ground truth.\n\n"
        "RULES:\n"
        "1. Judge by meaning, not exact string match. Ignore case, "
        "punctuation, accents, leading articles, and word order. Treat a "
        "reversed 'Last First' vs 'First Last' name order as a match — e.g. "
        "predicted 'Leonardo da Vinci' matches true 'da Vinci Leonardo', and "
        "predicted 'Starry Night' matches true 'The Starry Night'.\n"
        "2. If a prediction is the literal string '(none)', return null for "
        "that field.\n\n"
        "Respond with ONLY a JSON object, no prose, no markdown fences:\n"
        '{"title_correct": <true|false|null>, '
        '"artist_correct": <true|false|null>}'
    )


_GRADE_USER_TEMPLATE = (
    "Predicted title: {pred_title}\nTrue title: {true_title}\n\n"
    "Predicted artist: {pred_artist}\nTrue artist: {true_artist}\n\n"
    "Are the predictions correct?"
)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _parse_obj(raw: str) -> dict:
    """Extract the outermost JSON object from a raw model response."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in response")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("parsed JSON is not an object")
    return data


def _clean_str(value: object) -> str | None:
    """Normalise a model-supplied field to a non-empty string, else None."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.lower() in {"none", "null", "(none)"}:
        return None
    return value


def _clean_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _meta_words(classification: list[dict]) -> list[str]:
    """Distinct META span words, in order of first appearance."""
    seen: dict[str, None] = {}
    for span in classification:
        if span.get("category") == "META":
            word = span.get("word")
            if isinstance(word, str) and word.strip():
                seen.setdefault(word.strip(), None)
    return list(seen)


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class MetaValidator:
    """vLLM-served instruct LLM for the extract + grade passes."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 16384,
        max_new_tokens: int = 256,
    ):
        self.extract_system = _build_extract_system_prompt()
        self.grade_system = _build_grade_system_prompt()
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        # Greedy: deterministic JSON extraction, reproducible output.
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def _chat_many(self, system: str, users: list[str]) -> list[str]:
        if not users:
            return []
        conversations = [
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            for user in users
        ]
        outputs = self.llm.chat(conversations, self.sampling, use_tqdm=True)
        return [o.outputs[0].text for o in outputs]

    def extract_many(
        self, descriptions: list[str], candidate_lists: list[list[str]]
    ) -> list[str]:
        users = [
            _EXTRACT_USER_TEMPLATE.format(
                description=desc,
                candidates="\n".join(f"- {c}" for c in cands),
            )
            for desc, cands in zip(descriptions, candidate_lists)
        ]
        return self._chat_many(self.extract_system, users)

    def grade_many(self, items: list[dict]) -> list[str]:
        users = [
            _GRADE_USER_TEMPLATE.format(
                pred_title=it["pred_title"] or "(none)",
                true_title=it["true_title"] or "(none)",
                pred_artist=it["pred_artist"] or "(none)",
                true_artist=it["true_artist"] or "(none)",
            )
            for it in items
        ]
        return self._chat_many(self.grade_system, users)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _list_classify_dirs(classify_root: Path) -> list[Path]:
    return sorted(
        p for p in classify_root.iterdir()
        if p.is_dir() and (p / "classification.json").exists()
    )


def _load_annotations_by_rank(annotations_path: Path) -> dict[int, dict]:
    rows = json.loads(annotations_path.read_text())
    return {row["rank"]: row for row in rows}


@click.command(context_settings={"show_default": True})
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_DATA_DIR,
    help="Dataset root containing classify/, describe/ and annotations.json.",
)
@click.option(
    "--classify-name", default="classify",
    help="Subfolder with the classify-phase spans to read META from.",
)
@click.option(
    "--describe-name", default="describe",
    help="Subfolder with the describe-phase answers (for the full text).",
)
@click.option(
    "--annotations-name", default="annotations.json",
    help="Ground-truth metadata file inside the dataset root.",
)
@click.option(
    "--out-name", default="validate_meta",
    help="Subfolder inside the dataset root to write validations into.",
)
@click.option(
    "--batch-size", type=int, default=256,
    help="Paintings per vLLM call. vLLM continuous-batches within a chunk; this "
    "also sets the checkpoint-to-disk granularity for resumability.",
)
@click.option("--max-new-tokens", type=int, default=256)
@click.option(
    "--max-model-len", type=int, default=16384,
    help="vLLM context window (prompt + output).",
)
@click.option(
    "--gpu-memory-utilization", type=float, default=0.9,
    help="Fraction of total GPU memory vLLM reserves. Lower it on a shared GPU.",
)
@click.option(
    "--model-name", default="Qwen/Qwen3-4B-Instruct-2507",
    help="Hugging Face model id for the (non-thinking) instruct model.",
)
@click.option(
    "--overwrite/--no-overwrite", default=False,
    help="If False, skip paintings that already have a validation.json.",
)
def main(
    data_dir: Path,
    classify_name: str,
    describe_name: str,
    annotations_name: str,
    out_name: str,
    batch_size: int,
    max_new_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    model_name: str,
    overwrite: bool,
):
    classify_root = data_dir / classify_name
    describe_root = data_dir / describe_name
    out_root = data_dir / out_name
    out_root.mkdir(parents=True, exist_ok=True)

    annotations = _load_annotations_by_rank(data_dir / annotations_name)

    dirs = _list_classify_dirs(classify_root)
    if not overwrite:
        dirs = [
            d for d in dirs
            if not (out_root / d.name / "validation.json").exists()
        ]

    if not dirs:
        click.echo("Nothing to do — all paintings already validated.")
        return

    click.echo(
        f"Validating META for {len(dirs)} paintings with "
        f"batch_size={batch_size}"
    )

    validator = MetaValidator(
        model_name=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
    )

    pbar = tqdm(total=len(dirs))
    for start in range(0, len(dirs), batch_size):
        batch_dirs = dirs[start : start + batch_size]

        # Gather inputs for this chunk.
        records: list[dict] = []
        for d in batch_dirs:
            rank = int(d.name.split("_", 1)[0])
            row = annotations.get(rank, {})
            classification = json.loads(
                (d / "classification.json").read_text()
            )
            describe_answer = json.loads(
                (describe_root / d.name / "answer.json").read_text()
            )
            records.append({
                "dir": d,
                "stem": d.name,
                "candidates": _meta_words(classification),
                "description": describe_answer.get("text", ""),
                "true_title": _clean_str(row.get("title")),
                "true_artist": _clean_str(row.get("artistName")),
            })

        # --- Stage 1: extract title/artist from the META candidates. ---
        extractable = [r for r in records if r["candidates"]]
        raws = validator.extract_many(
            [r["description"] for r in extractable],
            [r["candidates"] for r in extractable],
        )
        for r, raw in zip(extractable, raws):
            r["extract_raw"] = raw
            try:
                obj = _parse_obj(raw)
                r["pred_title"] = _clean_str(obj.get("title"))
                r["pred_artist"] = _clean_str(obj.get("artist"))
            except (ValueError, json.JSONDecodeError) as exc:
                tqdm.write(f"[extract-failed] {r['stem']}: {exc}")
                r["pred_title"] = None
                r["pred_artist"] = None
        for r in records:
            r.setdefault("pred_title", None)
            r.setdefault("pred_artist", None)

        # --- Stage 2: grade the predictions against the ground truth. ---
        gradable = [
            r for r in records
            if r["pred_title"] is not None or r["pred_artist"] is not None
        ]
        raws = validator.grade_many(gradable)
        for r, raw in zip(gradable, raws):
            r["grade_raw"] = raw
            try:
                obj = _parse_obj(raw)
                r["title_correct"] = _clean_bool(obj.get("title_correct"))
                r["artist_correct"] = _clean_bool(obj.get("artist_correct"))
            except (ValueError, json.JSONDecodeError) as exc:
                tqdm.write(f"[grade-failed] {r['stem']}: {exc}")
                r["title_correct"] = None
                r["artist_correct"] = None
        for r in records:
            r.setdefault("title_correct", None)
            r.setdefault("artist_correct", None)

        # A prediction is only correct if it was actually made.
        for r in records:
            if r["pred_title"] is None:
                r["title_correct"] = None
            if r["pred_artist"] is None:
                r["artist_correct"] = None

            out_dir = out_root / r["stem"]
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "validation.json").write_text(json.dumps({
                "stem": r["stem"],
                "meta_candidates": r["candidates"],
                "title_prediction": r["pred_title"],
                "artist_prediction": r["pred_artist"],
                "ground_truth": {
                    "title": r["true_title"],
                    "artist": r["true_artist"],
                },
                "title_correct": r["title_correct"],
                "artist_correct": r["artist_correct"],
            }, indent=2))

        pbar.update(len(batch_dirs))
    pbar.close()


if __name__ == "__main__":
    main()
