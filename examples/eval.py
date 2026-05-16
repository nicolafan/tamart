"""Quantitative evaluation script using TAMExplainer.

Usage:
    python examples/eval.py [model_name] [dataset_path] [vis_path (optional)]

Example:
    python examples/eval.py Qwen/Qwen2-VL-2B-Instruct data/coco2014
"""
import json
import os
import random
import sys
import warnings

from tqdm import tqdm

from tamart.tam import TAMExplainer
from tamart.tam.eval import evaluate


random.seed(1024)
warnings.filterwarnings("ignore")


COCO_CATEGORY = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7,
    'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13,
    'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20,
    'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27,
    'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35,
    'snowboard': 36, 'ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40,
    'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44, 'glass': 46, 'cup': 47,
    'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54,
    'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60,
    'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67,
    'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76,
    'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82,
    'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89,
    'toothbrush': 90,
}


def prepare_input(dataset_path, processed_input=""):
    """Prepare (image, prompt, captions, mask, category) tuples for evaluation."""
    if processed_input:
        return json.load(open(os.path.join(dataset_path, processed_input)))

    input_data = []
    if 'coco' in dataset_path:
        seg_anno = json.load(open(os.path.join(dataset_path, 'annotations/instances_minival2014.json')))
        cap_anno = json.load(open(os.path.join(dataset_path, 'annotations/captions_val2014.json')))
        default_prompt = 'Write a one-sentence caption for this image:'
        cap_dic = {}
        for _ in cap_anno['annotations']:
            cap_dic.setdefault(_['image_id'], []).append(_['caption'])
        for _ in seg_anno['images']:
            fn = str(_['id']).zfill(12)
            input_data.append([
                os.path.join(dataset_path, 'image', fn + '.jpg'),
                default_prompt,
                cap_dic[_['id']],
                os.path.join(dataset_path, 'seg_label', fn + '.png'),
                COCO_CATEGORY,
            ])
    elif 'GranDf' in dataset_path or 'OpenPSG' in dataset_path:
        data = json.load(open(os.path.join(dataset_path, 'anno.json')))
        if 'GranDf' in dataset_path:
            default_prompt = 'Write a description for this image using around two sentences:'
        else:
            default_prompt = 'Write a description for this image using around three sentences:'
        for _ in data:
            input_data.append([
                os.path.join(dataset_path, _[0]),
                default_prompt,
                [_[1]],
                os.path.join(dataset_path, _[2]),
                _[3],
            ])

    return input_data


def main():
    model_name = sys.argv[1]
    dataset_path = sys.argv[2]
    vis_path = sys.argv[3] if len(sys.argv) > 3 else ""

    if 'Qwen' not in model_name:
        print("This script currently only supports Qwen2-VL models.")
        return

    input_data = prepare_input(dataset_path)
    explainer = TAMExplainer(model_name=model_name)

    results = []
    for sample_id, (img, prompt, caption, mask, category) in enumerate(tqdm(input_data, unit='sample')):
        save_dir = None
        if vis_path:
            base = img[0].split('/')[-2] if isinstance(img, list) else img.split('/')[-1].split('.')[0]
            save_dir = os.path.join(vis_path, f"{sample_id}_{base}")

        result = explainer.explain(img, prompt, save_dir=save_dir)
        generated_ids_trimmed = [result["tokens"]]
        metrics = evaluate(result["maps"], generated_ids_trimmed, explainer.processor, caption, mask, category)
        results.append(metrics)

    res = []
    for i in range(len(results[0])):
        values = []
        for _ in results:
            values.extend(_[i])
        res.append(sum(values) / len(values))

    print('Obj-IoU: %f, Func-IoU: %f, F1-IoU: %f, ROUGE-L: %f, METEOR: %f, Precision: %f, Recall: %f'
          % (res[0], res[1], 2 * res[0] * res[1] / (res[0] + res[1]), res[2], res[3], res[4], res[5]))


if __name__ == "__main__":
    main()
