import os

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .core import TAM
from .qwen_utils import process_vision_info


_QWEN2VL_SPECIAL_IDS = {
    'img_id': [151652, 151653],
    'prompt_id': [151653, [151645, 198, 151644, 77091]],
    'answer_id': [[198, 151644, 77091, 198], -1],
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
        self.processor = AutoProcessor.from_pretrained(model_name)
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
            content = [{"type": "video", "video": image}, {"type": "text", "text": prompt}]
        else:
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
                inputs['video_grid_thw'][0, 0],
                inputs['video_grid_thw'][0, 1] // 2,
                inputs['video_grid_thw'][0, 2] // 2,
            )
            vis_inputs = [[video_inputs[0][i] for i in range(0, len(video_inputs[0]))]]
        else:
            vision_shape = (
                inputs['image_grid_thw'][0, 1] // 2,
                inputs['image_grid_thw'][0, 2] // 2,
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
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        text_out = self.processor.batch_decode(
            generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return {
            "text": text_out,
            "maps": maps,
            "tokens": generated_trimmed[0],
        }
