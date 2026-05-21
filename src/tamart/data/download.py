"""Download the top-N most famous paintings from WikiArt.

Uses the WikiArt v2 JSON API. v2 is the version that supports paginating the
``MostViewedPaintings`` ranking via ``paginationToken``; v1's
``/App/Painting/MostViewedPaintings`` returns a single fixed batch (~600 items)
and ignores ``?page=``, so it can't reliably walk the ranking.

Output layout (under ``--output-dir``)::

    images/<rank>_<slug>.<ext>   highest available resolution per painting
    annotations.json             ordered list of full per-painting metadata
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import click
import requests
from tqdm import tqdm


WIKIART_BASE = "https://www.wikiart.org"
MOST_VIEWED_URL = f"{WIKIART_BASE}/en/api/2/MostViewedPaintings"
PAINTING_DETAIL_URL = f"{WIKIART_BASE}/en/api/2/Painting"

# Image size suffixes WikiArt serves, in decreasing order of resolution.
# Stripping every "!Variant.ext" suffix yields the original upload; if that
# 404s we fall back through progressively smaller variants.
IMAGE_VARIANTS: tuple[str, ...] = ("", "!HD.jpg", "!HalfHD.jpg", "!Large.jpg")

# WikiArt rate-limits aggressive clients (~10 req / 5 sec per the docs);
# a small fixed delay between metadata calls keeps us comfortably under that.
METADATA_REQUEST_DELAY = 0.2
REQUEST_TIMEOUT = 60


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "tamart-download/0.1 "
                "(+https://github.com/nicolafan/TAM; research use)"
            ),
            "Accept": "application/json",
        }
    )
    return s


def fetch_most_viewed(session: requests.Session, top_n: int) -> list[dict[str, Any]]:
    """Page through ``MostViewedPaintings`` until we have ``top_n`` items.

    Stops early if the API signals ``hasMore=False``.
    """
    items: list[dict[str, Any]] = []
    pagination_token: str | None = None
    pbar = tqdm(total=top_n, desc="Listing top paintings", unit="painting")
    try:
        while len(items) < top_n:
            params: dict[str, str] = {}
            if pagination_token:
                # WikiArt returns paginationToken already percent-encoded (raw
                # value is base64 with /, +, =). requests.params re-encodes,
                # producing %25xx which the server 500s on — so decode once
                # and let requests re-encode it exactly once.
                params["paginationToken"] = unquote(pagination_token)
            resp = session.get(MOST_VIEWED_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            batch = payload.get("data") or []
            if not batch:
                break
            for entry in batch:
                items.append(entry)
                pbar.update(1)
                if len(items) >= top_n:
                    break
            if not payload.get("hasMore"):
                break
            pagination_token = payload.get("paginationToken")
            if not pagination_token:
                break
            time.sleep(METADATA_REQUEST_DELAY)
    finally:
        pbar.close()
    return items[:top_n]


def fetch_painting_detail(
    session: requests.Session, painting_id: str
) -> dict[str, Any] | None:
    """Fetch the rich PaintingJson record for one painting id.

    Returns ``None`` on failure so a single bad record doesn't abort the run.
    """
    try:
        resp = session.get(
            PAINTING_DETAIL_URL,
            params={"id": painting_id},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _slugify(value: str) -> str:
    return _SLUG_RE.sub("-", value).strip("-").lower() or "painting"


def _strip_variant_suffix(image_url: str) -> str:
    """Remove a trailing ``!Variant.ext`` token to recover the original URL.

    WikiArt URLs look like ``.../the-starry-night-1889.jpg!Large.jpg``; the
    portion after the last ``!`` is the resized variant marker. Removing it
    yields the original upload, which is the largest resolution available.
    """
    last_bang = image_url.rfind("!")
    if last_bang == -1:
        return image_url
    return image_url[:last_bang]


def _candidate_image_urls(image_url: str) -> list[str]:
    """Build the resolution fallback list for a given API image URL."""
    base = _strip_variant_suffix(image_url)
    candidates = [base if v == "" else f"{base}{v}" for v in IMAGE_VARIANTS]
    # Drop duplicates (can happen when the original was already a variant we'd try).
    seen: set[str] = set()
    deduped: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _ext_from_url(url: str) -> str:
    path = urlparse(url).path
    _, dot, ext = path.rpartition(".")
    if not dot or len(ext) > 5 or not ext.isalnum():
        return "jpg"
    return ext.lower()


def download_image(
    session: requests.Session,
    image_url: str,
    target_dir: Path,
    filename_stem: str,
    overwrite: bool,
) -> tuple[Path | None, str | None]:
    """Try image variants in resolution order and write the first that works.

    Returns ``(saved_path, source_url)``. Both ``None`` if nothing was saved.
    """
    for candidate in _candidate_image_urls(image_url):
        ext = _ext_from_url(candidate)
        target = target_dir / f"{filename_stem}.{ext}"
        if target.exists() and not overwrite:
            return target, candidate
        try:
            with session.get(candidate, stream=True, timeout=REQUEST_TIMEOUT) as r:
                if r.status_code != 200:
                    continue
                tmp = target.with_suffix(target.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, target)
            return target, candidate
        except requests.RequestException:
            continue
    return None, None


def _build_annotation(
    rank: int,
    listing: dict[str, Any],
    detail: dict[str, Any] | None,
    image_path: Path | None,
    image_source_url: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    merged.update(listing)
    if detail:
        merged.update(detail)
    artist_url = merged.get("artistUrl")
    painting_url = merged.get("url")
    wikiart_page = (
        f"{WIKIART_BASE}/en/{artist_url}/{painting_url}"
        if artist_url and painting_url
        else None
    )
    return {
        "rank": rank,
        "id": merged.get("id"),
        "title": merged.get("title"),
        "artistName": merged.get("artistName"),
        "artistUrl": artist_url,
        "artistId": merged.get("artistId"),
        "url": painting_url,
        "completitionYear": merged.get("completitionYear"),
        "yearAsString": merged.get("yearAsString"),
        "period": merged.get("period"),
        "serie": merged.get("serie"),
        "genres": merged.get("genres"),
        "styles": merged.get("styles"),
        "media": merged.get("media"),
        "location": merged.get("location"),
        "galleries": merged.get("galleries"),
        "sizeX": merged.get("sizeX"),
        "sizeY": merged.get("sizeY"),
        "diameter": merged.get("diameter"),
        "tags": merged.get("tags"),
        "description": merged.get("description"),
        "width": merged.get("width"),
        "height": merged.get("height"),
        "apiImageUrl": merged.get("image"),
        "downloadedImageUrl": image_source_url,
        "imagePath": (
            str(image_path.relative_to(output_dir)) if image_path else None
        ),
        "wikiartPageUrl": wikiart_page,
    }


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Download the top-N most famous paintings from WikiArt (v2 API).\n\n"
        "Images go to <OUTPUT_DIR>/images/ at the highest resolution available; "
        "annotations.json holds the ordered metadata."
    ),
)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "-n",
    "--top-n",
    type=click.IntRange(min=1),
    default=1000,
    show_default=True,
    help="How many of the most-viewed paintings to fetch (in ranked order).",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Re-download images and re-fetch metadata even if files already exist.",
)
@click.option(
    "--annotations-name",
    default="annotations.json",
    show_default=True,
    help="Filename for the JSON annotations written under OUTPUT_DIR.",
)
def main(
    output_dir: Path,
    top_n: int,
    overwrite: bool,
    annotations_name: str,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = output_dir / annotations_name

    session = _make_session()
    click.echo(f"Fetching top {top_n} most-viewed paintings from WikiArt v2 API...")
    listings = fetch_most_viewed(session, top_n)
    if not listings:
        raise click.ClickException("No paintings returned by the WikiArt API.")
    click.echo(f"Got {len(listings)} listings; downloading details + images...")

    annotations: list[dict[str, Any]] = []
    pad_width = max(4, len(str(len(listings))))

    for rank, listing in enumerate(
        tqdm(listings, desc="Downloading", unit="painting"), start=1
    ):
        painting_id = listing.get("id")
        detail = fetch_painting_detail(session, painting_id) if painting_id else None
        time.sleep(METADATA_REQUEST_DELAY)

        merged_for_url = {**listing, **(detail or {})}
        image_url = merged_for_url.get("image")
        slug_source = merged_for_url.get("url") or painting_id or f"item-{rank}"
        stem = f"{rank:0{pad_width}d}_{_slugify(str(slug_source))}"

        saved_path: Path | None = None
        source_url: str | None = None
        if image_url:
            saved_path, source_url = download_image(
                session, image_url, images_dir, stem, overwrite
            )
            if saved_path is None:
                tqdm.write(
                    f"[warn] rank {rank} ({merged_for_url.get('title')!r}): "
                    f"no image variant downloaded from {image_url}"
                )

        annotations.append(
            _build_annotation(
                rank=rank,
                listing=listing,
                detail=detail,
                image_path=saved_path,
                image_source_url=source_url,
                output_dir=output_dir,
            )
        )

        # Persist incrementally so a crash doesn't lose accumulated metadata.
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

    click.echo(
        f"Done. {len(annotations)} annotations written to {annotations_path}; "
        f"images under {images_dir}."
    )


if __name__ == "__main__":
    main()
