"""Experiment: classify the description tokens of each painting with Qwen3.

The ``describe`` phase (:mod:`tamart.experiments.describe`) wrote, for every
painting, an ``answer.json`` containing the generated caption together with the
per-token ids and labels used by TAM. Here we feed that tokenized caption to a
text-only instruct LLM (``Qwen/Qwen3-4B-Instruct-2507`` by default), served
with **vLLM**, and ask it to group the tokens into contiguous, meaningful
expressions and assign each a category.

The caption is presented to the model as a numbered token list, e.g.::

    0: This
    1: ·image
    2: ·is
    ...

where a leading ``·`` marks the start of a new word (a token with a preceding
space) and tokens *without* it continue the previous word (sub-tokens). The
model answers with a JSON list of spans referring back to those indices::

    [
      {"tokens": [25], "word": "woman", "category": "CVO"},
      {"tokens": [70, 71], "word": "Renaissance period", "category": "STYLE"}
    ]

The parsed list is written to
``data/datasets/wikiart_most_viewed/classify/<stem>/classification.json`` —
the same per-painting subfolder layout as ``describe`` — alongside the raw
model output for debugging. The token indices are kept consistent with the
``token_labels`` in the corresponding ``describe`` answer so a notebook can map
a span straight onto its TAM activation maps.

vLLM batches all captions in a chunk concurrently (continuous batching);
``--batch-size`` is the chunk size, which also sets how often results are
checkpointed to disk for resumability.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Category name -> definition. Shared by the prompt (so the model is told what
# each label means) and by validation (so we can drop hallucinated labels).
CATEGORIES: dict[str, str] = {
    "CVO": "Concrete visual object with a physical location in the image "
    "(a depicted thing or its visible attribute, e.g. 'woman', "
    "'long flowing hair', 'body of water').",
    "ICON": "Named iconographic subject depicted in the scene — a "
    "mythological, religious, or historical figure (e.g. 'Dante', "
    "'Madonna', 'the Devil').",
    "STYLE": "Painterly or formal attribute: brushwork, palette, technique, "
    "or period/movement name. Capture only the core attribute phrase, never "
    "the explanation around it (e.g. 'loose brushwork', 'Renaissance', "
    "'vibrant colors', 'dynamic brushstrokes', 'hallucinatory colors').",
    "AFFECT": "Affective or interpretive claim — mood, emotion, or atmosphere. "
    "Capture only the bare mood word or short phrase (e.g. 'dramatic "
    "atmosphere', 'serene', 'iconic smile', 'sense of movement').",
    "META": "Artist name, painting title, or provenance metadata (e.g. "
    "'Mona Lisa', 'Leonardo da Vinci', '1642').",
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "datasets" / "wikiart_most_viewed"

# A label that is a model special token, e.g. "<|endoftext|>". These pad the
# tail of the generated answer and must not be offered for classification.
_SPECIAL_LABEL_RE = re.compile(r"^<\|.*\|>$")

# Hard cap on span length. A "meaningful expression" is a short phrase; spans
# longer than this are invariably whole clauses the model failed to trim
# (see _build_system_prompt rules 1/1b), so we drop them outright.
_MAX_SPAN_TOKENS = 10

# Purely positional/compositional filler. SPATIAL was dropped as a category,
# but the model still occasionally emits these (often mislabeled CVO); discard
# any span whose surface text is one of them.
_FILLER_PHRASES = frozenset({
    "foreground", "background", "middle", "center", "centre",
    "in the foreground", "in the background", "in the middle",
    "in the center", "in the centre", "in the distance",
    "at the bottom", "at the top", "on the left", "on the right",
    "to the left", "to the right", "in the corner",
})


def _is_filler(word: str) -> bool:
    return re.sub(r"[^\w\s]", "", word).lower().strip() in _FILLER_PHRASES

_USER_TEMPLATE = (
    "Classify the tokens of this painting description.\n\nTokens:\n{listing}"
)


def _build_system_prompt() -> str:
    cats = "\n".join(f"- {name}: {desc}" for name, desc in CATEGORIES.items())
    return (
        "You are an expert annotator of art descriptions. You are given a "
        "painting description that has been split into tokens, presented as a "
        "numbered list (one token per line, in the form 'index: token').\n\n"
        "TOKEN NOTATION: a leading '·' marks the START of a new word (the "
        "token had a preceding space in the original text). A token WITHOUT a "
        "leading '·' is a sub-token that continues the previous word — e.g. "
        "'St' + 'arry' = 'Starry'. When you build an expression, include every "
        "token index that spells it (sub-tokens included) and write the clean "
        "surface text (no '·') in 'word'.\n\n"
        "GOAL: pick out the most salient, MINIMAL expressions a viewer would "
        "tag in the image — the concrete things, named figures, titles, and "
        "style/mood attributes. Aim for about 10-20 expressions per "
        "description (fewer for short ones). Favour quality over coverage: "
        "select the expressions that carry real meaning and skip the rest.\n\n"
        "WHAT AN EXPRESSION IS: a single noun phrase, named entity, or "
        "attribute phrase, kept as SHORT as possible — the head noun plus only "
        "its essential adjectives. Good granularity: 'Starry Night', "
        "'small village or town', 'church', 'some trees', 'swirling stars', "
        "'vibrant, bold colors', 'dynamic brushstrokes', 'Vincent van Gogh'.\n\n"
        "Each expression must fit one of these categories:\n"
        f"{cats}\n\n"
        "CRITICAL RULES:\n"
        "1. NEVER emit a whole sentence or clause. If a candidate span "
        "contains a verb (e.g. 'depicts', 'creating', 'is dominated by', 'are "
        "characteristic of'), it is too long — cut it down to the bare noun "
        "phrase(s) inside it.\n"
        "1b. Keep every span SHORT — at most ~4 words. This is strict for "
        "STYLE and AFFECT: capture only the core attribute and DROP any "
        "trailing 'that …' / 'with …' / 'and the way …' explanation. E.g. "
        "from 'intense, almost hallucinatory colors and brushstrokes that "
        "convey a sense of movement and energy' emit 'hallucinatory colors' "
        "(STYLE) and 'sense of movement' (AFFECT), nothing more; from 'iconic "
        "smile and the way it captures the viewer's attention with its "
        "captivating and enigmatic expression' emit 'iconic smile' (AFFECT) "
        "and 'enigmatic expression' (AFFECT). If such a claim cannot be "
        "reduced to a short phrase, omit it entirely.\n"
        "2. Split coordinated or compound descriptions into one span per "
        "thing. E.g. from 'includes a small village or town at the bottom, "
        "with a church and some trees in the foreground' emit THREE spans: "
        "'small village or town', 'church', 'some trees'.\n"
        "3. Drop uninteresting filler — do NOT emit it as its own span and do "
        "NOT append it to a noun phrase: positional phrases ('in the "
        "foreground', 'at the bottom', 'on the left'), leading articles, "
        "connectives, verbs, and generic framing words ('image', 'painting', "
        "'scene') unless they belong to a title or proper name.\n"
        "4. Each expression maps to a contiguous run of token indices, in "
        "order, including sub-token continuations. Do not include leading or "
        "trailing punctuation / space-only tokens that are not part of the "
        "word.\n"
        "5. Spans must not overlap; assign each token to at most one span.\n"
        f"6. 'category' must be exactly one of: {', '.join(CATEGORIES)}.\n\n"
        "Respond with ONLY a JSON array, no prose, no markdown fences. Each "
        "element is an object with keys 'tokens' (list of integer indices), "
        "'word' (the clean surface expression), and 'category'. For example, "
        "for 'The painting titled \"Starry Night\" by Vincent van Gogh depicts "
        "swirling stars over a small village' a correct answer looks like:\n"
        '[{"tokens": [4, 5], "word": "Starry Night", "category": "META"}, '
        '{"tokens": [8, 9, 10], "word": "Vincent van Gogh", "category": '
        '"META"}, {"tokens": [13, 14], "word": "swirling stars", "category": '
        '"CVO"}, {"tokens": [18, 19], "word": "small village", "category": '
        '"CVO"}]'
    )


def _list_describe_dirs(describe_root: Path) -> list[Path]:
    return sorted(
        p for p in describe_root.iterdir()
        if p.is_dir() and (p / "answer.json").exists()
    )


def _format_token_listing(token_labels: list[str]) -> str:
    """Render the per-token numbered list shown to the model.

    Special tokens (e.g. ``<|endoftext|>`` padding) are skipped, but the
    original index of every kept token is preserved so the model's spans line
    up with the ``token_labels`` indexing in the describe answer.
    """
    lines = [
        f"{i}: {label}"
        for i, label in enumerate(token_labels)
        if not _SPECIAL_LABEL_RE.match(label)
    ]
    return "\n".join(lines)


def _parse_spans(raw: str, n_tokens: int) -> list[dict]:
    """Extract and validate the JSON span list from a raw model response."""
    # Strip markdown fences if present, then grab the outermost array.
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in response")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("parsed JSON is not a list")

    spans: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        toks = item.get("tokens")
        word = item.get("word")
        cat = item.get("category")
        if not isinstance(toks, list) or not toks:
            continue
        if len(toks) > _MAX_SPAN_TOKENS:
            continue
        if cat not in CATEGORIES or not isinstance(word, str):
            continue
        if _is_filler(word):
            continue
        if not all(isinstance(t, int) and 0 <= t < n_tokens for t in toks):
            continue
        spans.append({"tokens": toks, "word": word, "category": cat})
    return spans


class TokenClassifier:
    """vLLM-served instruct LLM for token-span classification."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 16384,
        max_new_tokens: int = 2048,
    ):
        self.system_prompt = _build_system_prompt()
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        # Greedy: this is a deterministic JSON-extraction task, so we want
        # reproducible output (the Instruct-2507 model is non-thinking, so the
        # "no greedy" caveat that applies to Qwen3 thinking mode is moot here).
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def classify_many(self, listings: list[str]) -> list[str]:
        """Return the raw model output for each token listing (order preserved).

        All listings are submitted at once; vLLM schedules them with continuous
        batching, so the GPU stays saturated regardless of per-caption length.
        """
        conversations = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": _USER_TEMPLATE.format(listing=lst)},
            ]
            for lst in listings
        ]
        outputs = self.llm.chat(conversations, self.sampling, use_tqdm=True)
        return [o.outputs[0].text for o in outputs]


@click.command(context_settings={"show_default": True})
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_DATA_DIR,
    help="Dataset root containing the describe-phase subfolder.",
)
@click.option(
    "--describe-name",
    default="describe",
    help="Subfolder with the describe-phase answers to read.",
)
@click.option(
    "--out-name",
    default="classify",
    help="Subfolder inside the dataset root to write classifications into.",
)
@click.option(
    "--batch-size", type=int, default=256,
    help="Captions per vLLM call. vLLM continuous-batches within a chunk; this "
    "also sets the checkpoint-to-disk granularity for resumability.",
)
@click.option("--max-new-tokens", type=int, default=2048)
@click.option(
    "--max-model-len", type=int, default=16384,
    help="vLLM context window (prompt + output).",
)
@click.option(
    "--gpu-memory-utilization", type=float, default=0.9,
    help="Fraction of total GPU memory vLLM reserves. Lower it on a shared GPU.",
)
@click.option(
    "--model-name",
    default="Qwen/Qwen3-4B-Instruct-2507",
    help="Hugging Face model id for the (non-thinking) instruct classifier.",
)
@click.option(
    "--overwrite/--no-overwrite", default=False,
    help="If False, skip paintings that already have a classification.json.",
)
def main(
    data_dir: Path,
    describe_name: str,
    out_name: str,
    batch_size: int,
    max_new_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    model_name: str,
    overwrite: bool,
):
    describe_root = data_dir / describe_name
    out_root = data_dir / out_name
    out_root.mkdir(parents=True, exist_ok=True)

    dirs = _list_describe_dirs(describe_root)
    if not overwrite:
        dirs = [
            d for d in dirs
            if not (out_root / d.name / "classification.json").exists()
        ]

    if not dirs:
        click.echo("Nothing to do — all paintings already classified.")
        return

    click.echo(
        f"Classifying {len(dirs)} descriptions with batch_size={batch_size}"
    )

    classifier = TokenClassifier(
        model_name=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
    )

    pbar = tqdm(total=len(dirs))
    for start in range(0, len(dirs), batch_size):
        batch_dirs = dirs[start : start + batch_size]
        answers = [
            json.loads((d / "answer.json").read_text()) for d in batch_dirs
        ]
        listings = [_format_token_listing(a["token_labels"]) for a in answers]
        raws = classifier.classify_many(listings)

        for d, answer, raw in zip(batch_dirs, answers, raws):
            out_dir = out_root / d.name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "raw_response.txt").write_text(raw)
            try:
                spans = _parse_spans(raw, n_tokens=len(answer["token_labels"]))
                (out_dir / "classification.json").write_text(
                    json.dumps(spans, indent=2)
                )
            except (ValueError, json.JSONDecodeError) as exc:
                tqdm.write(f"[parse-failed] {d.name}: {exc}")
        pbar.update(len(batch_dirs))
    pbar.close()


if __name__ == "__main__":
    main()
