import os

import cv2
import numpy as np
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .core import TAM
from .qwen_utils import process_vision_info

_QWEN2VL_SPECIAL_IDS = {
    "img_id": [151652, 151653],
    "prompt_id": [151653, [151645, 198, 151644, 77091]],
    "answer_id": [[198, 151644, 77091, 198], -1],
}


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
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
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

        Nothing is written to disk. Click a generated token to view its
        activation map overlaid on the input image; adjust colormap, alpha,
        and value range to inspect the relative scores.

        The title shows ``img_share = img_sum / (img_sum + txt_sum)`` — the
        fraction of total raw (post-ECI) attention that landed on image
        tokens vs. text tokens for the selected step. Useful when a map
        looks dim/blue: low share ⇒ model leaned on text, high share with
        low max ⇒ diffuse image attention.

        Args:
            image: path/PIL.Image for a single image (videos / multi-image
                not supported in interactive mode).
            prompt: text prompt for the MLLM.
            max_new_tokens: generation cap.
            figsize: matplotlib figure size for the image display.

        Returns:
            dict with the same shape as :meth:`explain` plus ``stats``: per-token
            ``{img_sum, txt_sum}`` dicts.
        """
        import ipywidgets as widgets
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
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
        stats_per_round = []
        raw_records = []
        for i in range(len(logits)):
            stats = {}
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
                stats,
            )
            maps.append(img_map)
            stats_per_round.append(stats)

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

        cmap_dd = widgets.Dropdown(
            options=[
                "jet",
                "turbo",
                "viridis",
                "plasma",
                "magma",
                "inferno",
                "hot",
                "cool",
                "RdBu_r",
            ],
            value="jet",
            description="Colormap:",
        )
        alpha_sl = widgets.FloatSlider(
            min=0.0,
            max=1.0,
            value=0.5,
            step=0.05,
            description="Alpha:",
            readout_format=".2f",
        )
        range_sl = widgets.IntRangeSlider(
            min=0,
            max=255,
            value=(0, 255),
            step=1,
            description="Range:",
            continuous_update=False,
        )
        autofit_btn = widgets.Button(
            description="Auto-fit range",
            tooltip="Set range to current map min/max",
        )
        token_sel = widgets.ToggleButtons(
            options=[(f"{i}|{l}", i) for i, l in enumerate(token_labels)],
            value=0,
            description="Token:",
            layout=widgets.Layout(flex_flow="row wrap", display="flex"),
        )

        out = widgets.Output()

        def redraw(_=None):
            with out:
                out.clear_output(wait=True)
                idx = token_sel.value
                raw = maps[idx]
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
                s = stats_per_round[idx]
                img_share = (
                    s["img_sum"] / (s["img_sum"] + s["txt_sum"] + 1e-8)
                    if s
                    else float("nan")
                )
                ax.set_title(
                    f"Token {idx}: {token_labels[idx]!r}"
                    f"   raw=[{int(raw.min())}, {int(raw.max())}]"
                    f"   img_share={img_share:.2f}"
                )
                ax.axis("off")
                sm = cm.ScalarMappable(
                    cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
                )
                sm.set_array([])
                fig.colorbar(sm, ax=ax, fraction=0.04)
                plt.tight_layout()
                plt.show()

        def autofit(_=None):
            raw = maps[token_sel.value]
            range_sl.value = (int(raw.min()), int(raw.max()))

        for w_ in [cmap_dd, alpha_sl, range_sl, token_sel]:
            w_.observe(redraw, names="value")
        autofit_btn.on_click(autofit)

        display(widgets.HTML(f"<b>Generated:</b> {text_out}"))
        display(
            widgets.VBox(
                [
                    widgets.HBox([cmap_dd, alpha_sl, autofit_btn]),
                    range_sl,
                    token_sel,
                    out,
                ]
            )
        )
        redraw()

        return {
            "text": text_out,
            "maps": maps,
            "tokens": trimmed,
            "stats": stats_per_round,
        }
