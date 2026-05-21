import json
import os
import re
from html import escape as _html_escape
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as st_load_file
from safetensors.torch import save_file as st_save_file
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .core import TAM
from .qwen_utils import process_vision_info

_QWEN2VL_SPECIAL_IDS = {
    "img_id": [151652, 151653],
    "prompt_id": [151653, [151645, 198, 151644, 77091]],
    "answer_id": [[198, 151644, 77091, 198], -1],
}

# Display colors for the classification categories produced by
# ``tamart.experiments.classify``. Used when ``explain_interactive_precomputed``
# is given a classify folder to render the caption spans. Unknown categories
# fall back to a neutral gray.
_CATEGORY_COLORS = {
    "CVO": "#1f77b4",
    "ICON": "#9467bd",
    "STYLE": "#2ca02c",
    "AFFECT": "#d62728",
    "META": "#ff7f0e",
}
_CATEGORY_FALLBACK_COLOR = "#7f7f7f"

# A label that is a model special token, e.g. "<|endoftext|>". These are skipped
# when reconstructing the caption text for the classify-span view.
_SPECIAL_LABEL_RE = re.compile(r"^<\|.*\|>$")


def _category_color(category: str) -> str:
    return _CATEGORY_COLORS.get(category, _CATEGORY_FALLBACK_COLOR)


class TAMExplainer:
    """High-level API for Token Activation Map explanations over Qwen2-VL.

    Loads the model + processor once on construction and exposes :meth:`explain`
    for inference + per-token activation maps, with optional visualization.

    Example:
        >>> explainer = TAMExplainer()
        >>> result = explainer.explain("imgs/demo.jpg", "Describe this image.")
        >>> print(result["text"])
        >>> # save visualizations
        >>> explainer.explain("imgs/demo.jpg", "Describe this image.", save_dir="out/")
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device_map: str = "auto",
        torch_dtype="auto",
    ):
        self.model_name = model_name
        # transformers >=4.56 renamed torch_dtype -> dtype.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, dtype=torch_dtype, device_map=device_map
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        # use_fast=False keeps the slow image processor: transformers >=4.56
        # defaults to the fast one, which preprocesses slightly differently and
        # would diverge from the describe data generated under the old default.
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=False
        )
        # left padding is required so that generated tokens always sit at the
        # tail of each batch row and `outputs.sequences[b, -k:]` is the answer.
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self._special_ids = _QWEN2VL_SPECIAL_IDS

    def explain(
        self,
        image,
        prompt: str,
        save_dir: str | None = None,
        max_new_tokens: int = 256,
    ):
        """Run TAM inference on an image (or list of video frames) with a prompt.

        Args:
            image: path/PIL.Image for a single image, or a list of paths/PIL.Images
                for video frames.
            prompt: text prompt for the MLLM.
            save_dir: if given, per-token visualization images are written here as
                ``{i}.jpg`` (one per generation round). The directory is created
                if missing.
            max_new_tokens: generation cap.

        Returns:
            dict with keys:
                - ``text``: generated answer string.
                - ``maps``: list of np.ndarray TAM maps (one per generation round).
                - ``tokens``: list of generated token ids (trimmed of the prompt).
        """
        is_video = isinstance(image, list)

        if is_video:
            content = [
                {"type": "video", "video": image},
                {"type": "text", "text": prompt},
            ]
        else:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        generated_ids = outputs.sequences
        logits = [self.model.lm_head(feats[-1]) for feats in outputs.hidden_states]

        if is_video:
            vision_shape = (
                inputs["video_grid_thw"][0, 0],
                inputs["video_grid_thw"][0, 1] // 2,
                inputs["video_grid_thw"][0, 2] // 2,
            )
            vis_inputs = [[video_inputs[0][i] for i in range(0, len(video_inputs[0]))]]
        else:
            vision_shape = (
                inputs["image_grid_thw"][0, 1] // 2,
                inputs["image_grid_thw"][0, 2] // 2,
            )
            vis_inputs = image_inputs

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        maps = []
        raw_map_records = []
        eval_only = save_dir is None
        for i in range(len(logits)):
            save_fn = os.path.join(save_dir, f"{i}.jpg") if save_dir is not None else ""
            img_map = TAM(
                generated_ids[0].cpu().tolist(),
                vision_shape,
                logits,
                self._special_ids,
                vis_inputs,
                self.processor,
                save_fn,
                i,
                raw_map_records,
                eval_only,
            )
            maps.append(img_map)

        generated_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        text_out = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {
            "text": text_out,
            "maps": maps,
            "tokens": generated_trimmed[0],
        }

    def explain_interactive(self, image, prompt, max_new_tokens=256, figsize=(8, 6)):
        """Run TAM inference and display an interactive widget in a notebook.

        Nothing is written to disk. Click one or more generated tokens to view
        their activation maps overlaid on the input image; when multiple
        tokens are selected, their maps are merged by taking the per-pixel
        mean. Adjust colormap, alpha, and value range to inspect the relative
        scores.

        Args:
            image: path/PIL.Image for a single image (videos / multi-image
                not supported in interactive mode).
            prompt: text prompt for the MLLM.
            max_new_tokens: generation cap.
            figsize: matplotlib figure size for the image display.

        Returns:
            dict with text, maps, tokens, token_labels, proc_img, and a
            ``selected_indices`` callable.
        """
        import ipywidgets as widgets
        from IPython.display import display

        if isinstance(image, list):
            raise NotImplementedError(
                "explain_interactive currently supports single images only"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        generated_ids = outputs.sequences
        logits = [self.model.lm_head(feats[-1]) for feats in outputs.hidden_states]

        vision_shape = (
            inputs["image_grid_thw"][0, 1] // 2,
            inputs["image_grid_thw"][0, 2] // 2,
        )

        maps = []
        raw_records = []
        for i in range(len(logits)):
            img_map = TAM(
                generated_ids[0].cpu().tolist(),
                vision_shape,
                logits,
                self._special_ids,
                image_inputs,
                self.processor,
                "",
                i,
                raw_records,
                True,
            )
            maps.append(img_map)

        trimmed = generated_ids[0][len(inputs.input_ids[0]) :].cpu().tolist()
        raw_labels = [self.processor.tokenizer.decode([t]) for t in trimmed]
        token_labels = [
            (l.replace("\n", "\\n").replace(" ", "·") or f"<{trimmed[i]}>")
            for i, l in enumerate(raw_labels)
        ]
        text_out = self.processor.batch_decode(
            [trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        proc_img = np.array(image_inputs[0])  # RGB uint8

        return self._render_token_viewer(
            maps=maps,
            token_labels=token_labels,
            tokens=trimmed,
            text_out=text_out,
            proc_img=proc_img,
            figsize=figsize,
        )

    def _make_overlay_controls(self):
        """Build the shared colormap / alpha / range / autofit widgets."""
        import ipywidgets as widgets

        cmap_dd = widgets.Dropdown(
            options=[
                "jet", "turbo", "viridis", "plasma", "magma",
                "inferno", "hot", "cool", "RdBu_r",
            ],
            value="jet",
            description="Colormap:",
        )
        alpha_sl = widgets.FloatSlider(
            min=0.0, max=1.0, value=0.5, step=0.05,
            description="Alpha:", readout_format=".2f",
        )
        range_sl = widgets.IntRangeSlider(
            min=0, max=255, value=(0, 255), step=1,
            description="Range:", continuous_update=False,
        )
        autofit_btn = widgets.Button(
            description="Auto-fit range",
            tooltip="Set range to current map min/max",
        )
        return cmap_dd, alpha_sl, range_sl, autofit_btn

    def _draw_overlay(self, out, raw, proc_img, title_head, cmap_dd,
                      alpha_sl, range_sl, figsize):
        """Render a single averaged map blended over the processed image."""
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        with out:
            out.clear_output(wait=True)
            if raw is None:
                print("Nothing selected.")
                return
            h, w = proc_img.shape[:2]
            full = cv2.resize(raw.astype(np.float32), (w, h))
            vmin, vmax = range_sl.value
            cmap = plt.get_cmap(cmap_dd.value)
            normed = np.clip((full - vmin) / max(vmax - vmin, 1e-6), 0, 1)
            colored = cmap(normed)[:, :, :3]
            alpha = alpha_sl.value
            blended = np.clip(
                (1 - alpha) * (proc_img / 255.0) + alpha * colored, 0, 1
            )
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(blended)
            ax.set_title(
                f"{title_head}   raw=[{int(raw.min())}, {int(raw.max())}]"
            )
            ax.axis("off")
            sm = cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.04)
            plt.tight_layout()
            plt.show()

    def _render_token_viewer(self, maps, token_labels, tokens, text_out,
                             proc_img, figsize):
        """Per-token toggle-button viewer (shared by live and precomputed)."""
        import ipywidgets as widgets
        from IPython.display import display

        cmap_dd, alpha_sl, range_sl, autofit_btn = self._make_overlay_controls()
        out = widgets.Output()

        token_btns = [
            widgets.ToggleButton(
                value=(i == 0),
                description=f"{i}|{token_labels[i]}",
                layout=widgets.Layout(width="auto"),
            )
            for i in range(len(token_labels))
        ]
        token_box = widgets.HBox(
            token_btns,
            layout=widgets.Layout(flex_flow="row wrap", display="flex"),
        )

        def selected_indices():
            return [i for i, b in enumerate(token_btns) if b.value]

        def merged_map(indices):
            return np.mean(np.stack([maps[i] for i in indices], axis=0), axis=0)

        def redraw(_=None):
            indices = selected_indices()
            if not indices:
                self._draw_overlay(
                    out, None, proc_img, "", cmap_dd, alpha_sl, range_sl, figsize
                )
                return
            raw = merged_map(indices)
            if len(indices) == 1:
                title_head = f"Token {indices[0]}: {token_labels[indices[0]]!r}"
            else:
                preview = ", ".join(f"{i}|{token_labels[i]}" for i in indices[:6])
                if len(indices) > 6:
                    preview += f", … (+{len(indices) - 6})"
                title_head = f"Tokens [{preview}] (mean)"
            self._draw_overlay(
                out, raw, proc_img, title_head, cmap_dd, alpha_sl,
                range_sl, figsize,
            )

        def autofit(_=None):
            indices = selected_indices()
            if not indices:
                return
            raw = merged_map(indices)
            range_sl.value = (int(raw.min()), int(raw.max()))

        for w_ in [cmap_dd, alpha_sl, range_sl]:
            w_.observe(redraw, names="value")
        for btn in token_btns:
            btn.observe(redraw, names="value")
        autofit_btn.on_click(autofit)

        display(widgets.HTML(f"<b>Generated:</b> {text_out}"))
        display(
            widgets.VBox(
                [
                    widgets.HBox([cmap_dd, alpha_sl, autofit_btn]),
                    range_sl,
                    widgets.Label(value="Tokens (click to view their maps):"),
                    token_box,
                    out,
                ]
            )
        )
        redraw()

        return {
            "text": text_out,
            "maps": maps,
            "tokens": tokens,
            "token_labels": token_labels,
            "proc_img": proc_img,
            "selected_indices": selected_indices,
        }

    def explain_batch(
        self,
        images,
        prompt,
        max_new_tokens: int = 256,
        compute_stats: bool = True,
    ):
        """Run TAM on a batch of images that share a single text prompt.

        Generation is performed in a single batched ``model.generate`` call;
        per-sample TAM maps are then computed by slicing the batched logits
        and tokens. Each image may have a different resolution — the
        processor handles per-sample ``image_grid_thw``.

        Args:
            images: list of paths/PIL.Images (single images only; no videos).
            prompt: shared text prompt.
            max_new_tokens: generation cap.
            compute_stats: if True, attach ``img_sum``/``txt_sum`` per token.

        Returns:
            list of dicts, one per input image, each with:
                - ``text``: generated answer
                - ``maps``: list of np.ndarray (uint8) per generated token
                - ``tokens``: list of generated token ids
                - ``token_labels``: per-token decoded strings
                - ``proc_img``: processed RGB uint8 image used by TAM
                - ``stats`` (if compute_stats): per-token img/txt sums
        """
        if not isinstance(images, (list, tuple)):
            images = [images]

        messages_list = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for img in images
        ]
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_list
        ]
        all_image_inputs = []
        for m in messages_list:
            imgs_in, _ = process_vision_info(m)
            all_image_inputs.extend(imgs_in)

        inputs = self.processor(
            text=texts,
            images=all_image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        generated_ids = outputs.sequences  # (B, prefill + new)
        B = generated_ids.shape[0]
        prefill_len = inputs["input_ids"].shape[1]

        # Per-step logits over the LM head. We compute them once and slice
        # per-sample later. Step 0 is (B, prefill_len, V); subsequent steps
        # are (B, 1, V).
        per_step_logits = [
            self.model.lm_head(feats[-1]) for feats in outputs.hidden_states
        ]

        # image_grid_thw is one row per image in the batch.
        grid_thw = inputs["image_grid_thw"]

        results = []
        for b in range(B):
            sample_tokens = generated_ids[b].cpu().tolist()
            sample_logits = [step[b : b + 1] for step in per_step_logits]
            vision_shape = (
                int(grid_thw[b, 1] // 2),
                int(grid_thw[b, 2] // 2),
            )
            vis_inputs = [all_image_inputs[b]]

            maps = []
            stats_per_round = []
            raw_records = []
            for i in range(len(sample_logits)):
                stats = {} if compute_stats else None
                img_map = TAM(
                    sample_tokens,
                    vision_shape,
                    sample_logits,
                    self._special_ids,
                    vis_inputs,
                    self.processor,
                    "",
                    i,
                    raw_records,
                    True,
                    stats,
                )
                maps.append(img_map)
                if stats is not None:
                    stats_per_round.append(stats)

            # The trimmed answer = tokens beyond the prefill length. With
            # left padding all batch rows share a prefill_len.
            trimmed = generated_ids[b, prefill_len:].cpu().tolist()
            raw_labels = [self.processor.tokenizer.decode([t]) for t in trimmed]
            token_labels = [
                (l.replace("\n", "\\n").replace(" ", "·") or f"<{trimmed[i]}>")
                for i, l in enumerate(raw_labels)
            ]
            text_out = self.processor.batch_decode(
                [trimmed],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            proc_img = np.array(all_image_inputs[b])

            results.append(
                {
                    "text": text_out,
                    "maps": maps,
                    "tokens": trimmed,
                    "token_labels": token_labels,
                    "proc_img": proc_img,
                    "stats": stats_per_round if compute_stats else None,
                }
            )

        return results

    @staticmethod
    def save_precomputed(out_dir, result):
        """Persist an :meth:`explain_batch` result to ``out_dir`` for later
        replay by :meth:`explain_interactive_precomputed`.

        Layout written:
            ``out_dir/answer.json``    — text, token ids, labels, stats
            ``out_dir/maps.safetensors`` — keys = token index strings, values =
                                            float32 maps in [0, 1]
            ``out_dir/proc_img.png``   — processed RGB image used for overlay
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tensor_dict = {}
        for i, m in enumerate(result["maps"]):
            arr = np.asarray(m).astype(np.float32) / 255.0
            tensor_dict[str(i)] = torch.from_numpy(arr).contiguous()
        st_save_file(tensor_dict, str(out_dir / "maps.safetensors"))

        Image.fromarray(result["proc_img"]).save(out_dir / "proc_img.png")

        answer = {
            "text": result["text"],
            "tokens": list(result["tokens"]),
            "token_labels": list(result["token_labels"]),
            "stats": result.get("stats"),
        }
        with open(out_dir / "answer.json", "w") as f:
            json.dump(answer, f, indent=2)

    def explain_interactive_precomputed(
        self, folder, classify_folder=None, figsize=(8, 6)
    ):
        """Replay a previously-saved annotation as an interactive widget.

        ``folder`` must be a directory written by :meth:`save_precomputed`
        (contains ``answer.json``, ``maps.safetensors``, ``proc_img.png``).
        No model inference is performed — maps, tokens, and the processed
        image are read from disk.

        Two viewing modes:

        * **token mode** (default): click one or more generated tokens to
          overlay their (mean) activation maps on the image.
        * **span mode** (``classify_folder`` given): the folder must contain a
          ``classification.json`` written by ``tamart.experiments.classify``.
          The full caption is shown with the classified multi-token spans
          highlighted by category; click a single span to view the mean
          activation map over its tokens.

        Args:
            folder: describe-phase folder with the precomputed maps.
            classify_folder: optional folder with ``classification.json`` for
                the same painting. If given, the span viewer is shown instead
                of the per-token one. Pass ``True`` to reuse ``folder``.
            figsize: matplotlib figure size for the image display.
        """
        folder = Path(folder)
        with open(folder / "answer.json") as f:
            answer = json.load(f)
        token_labels = answer["token_labels"]
        text_out = answer["text"]

        tensors = st_load_file(str(folder / "maps.safetensors"))
        maps = [
            (tensors[str(i)].numpy() * 255.0).astype(np.float32)
            for i in range(len(token_labels))
        ]

        proc_img = np.array(Image.open(folder / "proc_img.png").convert("RGB"))

        if classify_folder is not None:
            classify_dir = folder if classify_folder is True else Path(classify_folder)
            with open(classify_dir / "classification.json") as f:
                spans = json.load(f)
            return self._render_span_viewer(
                maps=maps,
                token_labels=token_labels,
                spans=spans,
                text_out=text_out,
                proc_img=proc_img,
                figsize=figsize,
            )

        return self._render_token_viewer(
            maps=maps,
            token_labels=token_labels,
            tokens=answer["tokens"],
            text_out=text_out,
            proc_img=proc_img,
            figsize=figsize,
        )

    def _render_span_viewer(self, maps, token_labels, spans, text_out,
                            proc_img, figsize):
        """Caption + classified-span viewer.

        Renders the full caption with each classified span highlighted by its
        category color; clicking a span overlays the mean activation map over
        that span's tokens.
        """
        import ipywidgets as widgets
        from IPython.display import display

        n_tokens = len(maps)
        # Keep only spans whose token indices are all in range.
        spans = [
            s for s in spans
            if s.get("tokens")
            and all(isinstance(t, int) and 0 <= t < n_tokens for t in s["tokens"])
        ]

        # token index -> owning span index (first span wins; classify spans
        # are non-overlapping by construction).
        tok2span = {}
        for si, s in enumerate(spans):
            for t in s["tokens"]:
                tok2span.setdefault(t, si)

        cmap_dd, alpha_sl, range_sl, autofit_btn = self._make_overlay_controls()
        out = widgets.Output()
        state = {"sel": 0 if spans else None}

        def caption_html(selected):
            parts = ["<div style='line-height:2.1; font-size:1.05em;'>"]
            i = 0
            while i < len(token_labels):
                si = tok2span.get(i)
                if si is None:
                    label = token_labels[i]
                    if not _SPECIAL_LABEL_RE.match(label):
                        surface = label.replace("·", " ").replace("\\n", " ")
                        parts.append(_html_escape(surface))
                    i += 1
                    continue
                s = spans[si]
                color = _category_color(s["category"])
                style = (
                    f"background:{color};color:#fff;padding:1px 5px;margin:0 1px;"
                    "border-radius:5px;cursor:default;"
                )
                if si == selected:
                    style += "outline:3px solid #111;font-weight:bold;"
                parts.append(
                    f'<span title="{_html_escape(s["category"])}" style="{style}">'
                    f'{_html_escape(s["word"])}'
                    f'<sub style="opacity:.85;font-size:.7em;"> '
                    f'{_html_escape(s["category"])}</sub></span>'
                )
                while i < len(token_labels) and tok2span.get(i) == si:
                    i += 1
            parts.append("</div>")
            return "".join(parts)

        legend = " ".join(
            f'<span style="background:{_category_color(c)};color:#fff;'
            f'padding:1px 6px;border-radius:5px;margin-right:4px;">{c}</span>'
            for c in _CATEGORY_COLORS
        )
        caption_widget = widgets.HTML()

        span_btns = [
            widgets.Button(
                description=f"{s['word']}  [{s['category']}]",
                tooltip=f"tokens {s['tokens']}",
                layout=widgets.Layout(width="auto", margin="2px"),
            )
            for s in spans
        ]
        for bi, b in enumerate(span_btns):
            b.style.button_color = _category_color(spans[bi]["category"])
        span_box = widgets.HBox(
            span_btns, layout=widgets.Layout(flex_flow="row wrap", display="flex")
        )

        def current_raw():
            if state["sel"] is None:
                return None
            idxs = spans[state["sel"]]["tokens"]
            return np.mean(np.stack([maps[t] for t in idxs], axis=0), axis=0)

        def refresh(_=None):
            caption_widget.value = caption_html(state["sel"])
            for bi, b in enumerate(span_btns):
                b.layout.border = (
                    "3px solid #111" if bi == state["sel"] else "1px solid #ccc"
                )
            if state["sel"] is None:
                title = "No span selected"
            else:
                s = spans[state["sel"]]
                title = f"{s['word']!r} [{s['category']}] · tokens {s['tokens']}"
            self._draw_overlay(
                out, current_raw(), proc_img, title, cmap_dd, alpha_sl,
                range_sl, figsize,
            )

        def make_click(bi):
            def handler(_):
                state["sel"] = bi
                refresh()
            return handler

        for bi, b in enumerate(span_btns):
            b.on_click(make_click(bi))

        def autofit(_=None):
            raw = current_raw()
            if raw is not None:
                range_sl.value = (int(raw.min()), int(raw.max()))

        for w_ in [cmap_dd, alpha_sl, range_sl]:
            w_.observe(refresh, names="value")
        autofit_btn.on_click(autofit)

        if not spans:
            display(widgets.HTML(
                "<b>No spans found in classification.json.</b>"
            ))

        display(
            widgets.VBox(
                [
                    widgets.HTML(f"<b>Legend:</b> {legend}"),
                    caption_widget,
                    widgets.HBox([cmap_dd, alpha_sl, autofit_btn]),
                    range_sl,
                    widgets.Label(value="Spans (click one to view its map):"),
                    span_box,
                    out,
                ]
            )
        )
        refresh()

        return {
            "text": text_out,
            "maps": maps,
            "token_labels": token_labels,
            "spans": spans,
            "proc_img": proc_img,
            "selected_span": lambda: state["sel"],
        }
