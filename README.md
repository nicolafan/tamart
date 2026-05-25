# TAMArt — project page

Interactive project page for **“Understanding How MLLMs Describe Artworks Using
Token Activation Maps.”** Served via GitHub Pages from this `gh-pages` branch.

It is a fully static site (no build step, no backend). Everything under
`assets/` is precomputed from the paper's pipeline outputs:

- `assets/data/index.json` — gallery list of paintings.
- `assets/data/paintings/<stem>.json` — caption, semantically-typed spans, and
  each span's Token Activation Map (base64 `uint8` over the vision grid).
- `assets/img/<stem>.jpg` — the painting.
- `assets/masks/<stem>__<span>.png` — SAM 3 mask for CVO/ICON spans.

The TAM heatmap, the Otsu threshold, and the SAM overlay are all rendered
client-side in `js/app.js` (the Otsu threshold is computed live in the browser).

## Regenerating the assets

`build_site_assets.py` (copied here for reference) reads the dataset on the
research machine and rebuilds the `assets/` tree:

```bash
# from the main project checkout, with the project venv
.venv/bin/python scripts/build_site_assets.py --n 150 --clean
```

Then copy the refreshed `_site/` contents onto this branch and push.
