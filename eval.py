#!/usr/bin/env python

import argparse
import copy
import json
import multiprocessing as mp
import os
import torch

from typing import Literal
from pycocotools.coco import COCO
from tqdm import tqdm

from utils.inference import (
    AbstractObjectDetector,
    DetectedObject,
    TextPrompt,
    VisualPrompt,
)


PATH_ROOT = os.path.realpath(os.path.dirname(__file__))
CLASS_INFO = {}

for data_name in ["dataset1", "dataset2", "dataset3"]:
    with open(f"{PATH_ROOT}/NTIRE2026/{data_name}/annotations/1_shot.json") as f:
        data = json.load(f)
        CLASS_INFO[data_name] = [
            cat_dict["name"] for cat_dict in sorted(data['categories'], key=lambda x: x['id'])
        ]

CFG = {
    ('dataset1', 1): {
        'type': 'codetr',
        'args': {
            'path_cfg': f'{PATH_ROOT}/config/bbox_config.py',
            'path_weights': os.path.join(PATH_ROOT, "weights", "codetr-dataset1-1.pth"),
            'class_names': CLASS_INFO['dataset1'],
            'use_tta': True,
        }
    },
    ('dataset1', 5): {
        'type': 'codetr',
        'args': {
            'path_cfg': f'{PATH_ROOT}/config/bbox_config.py',
            'path_weights': os.path.join(PATH_ROOT, "weights", "codetr-dataset1-5.pth"),
            'class_names': CLASS_INFO['dataset1'],
            'use_tta': True,
        }
    },
    ('dataset1', 10): {
        'type': 'codetr',
        'args': {
            'path_cfg': f'{PATH_ROOT}/config/bbox_config.py',
            'path_weights': os.path.join(PATH_ROOT, "weights", "codetr-dataset1-10.pth"),
            'class_names': CLASS_INFO['dataset1'],
            'use_tta': False,
        }
    },

    ('dataset2', 1): {
        'type': 'codetr',
        'args': {
            'path_cfg': f'{PATH_ROOT}/config/bbox_config.py',
            'path_weights': os.path.join(PATH_ROOT, "weights", "codetr-dataset2-1.pth"),
            'class_names': CLASS_INFO['dataset2'],
            'use_tta': True,
        }
    },
    ('dataset2', 5): {
        'type': 'codetr',
        'args': {
            'path_cfg': f'{PATH_ROOT}/config/bbox_config.py',
            'path_weights': os.path.join(PATH_ROOT, "weights", "codetr-dataset2-5.pth"),
            'class_names': CLASS_INFO['dataset2'],
            'use_tta': True,
        }
    },
    ('dataset2', 10): {
        'type': 'codetr',
        'args': {
            'path_cfg': f'{PATH_ROOT}/config/bbox_config.py',
            'path_weights': os.path.join(PATH_ROOT, "weights", "codetr-dataset2-10.pth"),
            'class_names': CLASS_INFO['dataset2'],
            'use_tta': True,
        }
    },

    ('dataset3', 1): {
        'type': 'zero',
        'args': {
            'model_endpoint': f'http://{os.environ["SERVER_IP"]}:8080/invocations',
            'prompt_type': 'visual',
        }
    },
    ('dataset3', 5): {
        'type': 'codetr',
        'args': {
            'path_cfg': f'{PATH_ROOT}/config/bbox_config.py',
            'path_weights': os.path.join(PATH_ROOT, "weights", "codetr-dataset3-5.pth"),
            'class_names': CLASS_INFO['dataset3'],
            'use_tta': False,
        }
    },
    ('dataset3', 10): {
        'type': 'codetr',
        'args': {
            'path_cfg': f'{PATH_ROOT}/config/bbox_config.py',
            'path_weights': os.path.join(PATH_ROOT, "weights", "codetr-dataset3-10.pth"),
            'class_names': CLASS_INFO['dataset3'],
            'use_tta': False,
        }
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_name",
        choices=["dataset1", "dataset2", "dataset3"],
    )
    parser.add_argument(
        "shot",
        type=int,
        choices=[1, 5, 10],
    )
    parser.add_argument(
        "--data-path",
        default='./NTIRE2026/',
    )
    parser.add_argument(
        "--output",
        default="/workspace/submit/",
        type=str,
    )
    parser.add_argument(
        '--zero-model-endpoint',
        default=None,
        type=str,
        help="Endpoint for the zero-shot model. If not provided, the default endpoint in CFG will be used.",
    )
    parser.add_argument(
        '--codetr-weights',
        default=None,
        type=str,
        help="Path to the weights for CODETR. If not provided, the default path in CFG will be used.",
    )
    return parser.parse_args()


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def prepare_visual_queries(data_root, k):
    coco = COCO(f"{data_root}/annotations/{k}_shot.json")
    queries = []
    
    for img_id in coco.getImgIds():
        img_dict = coco.loadImgs([img_id])[0]
        ann_dict_lst = coco.loadAnns(coco.getAnnIds(img_id))

        for ann_dict in ann_dict_lst:
            text = coco.cats[ann_dict["category_id"]]["name"]
            bbox = ann_dict["bbox"]
            queries.append(VisualPrompt(
                class_name=text,
                bbox_xywh=bbox,
                image_path=os.path.join(data_root, "train", img_dict["file_name"]),
            ))
    return queries


def prepare_text_queries(data_root, k):
    coco = COCO(f"{data_root}/annotations/{k}_shot.json")
    queries = []

    for cat_dict in sorted(coco.cats.values(), key=lambda x: x['id']):
        queries.append(TextPrompt(
            class_name=cat_dict["name"],
        ))
    return queries


def main():
    args = parse_args()

    data_root = os.path.join(args.data_path, args.data_name)
    os.environ["DATA_ROOT"] = data_root

    gt_json = os.path.join(data_root, "annotations", "test.json")
    coco = COCO(gt_json)
    label2id = {cat_dict["name"]: cat_dict["id"] for cat_dict in coco.cats.values()}

    model_cfg = CFG.get((args.data_name, args.shot), None)
    if model_cfg is None:
        raise ValueError(
            f"No model configuration found for dataset {args.data_name} with {args.shot}-shot."
        )

    model_cfg = copy.deepcopy(model_cfg)

    if model_cfg["type"] == "zero" and args.zero_model_endpoint is not None:
        model_cfg["args"]["model_endpoint"] = args.zero_model_endpoint
    if model_cfg["type"] == "codetr" and args.codetr_weights is not None:
        model_cfg["args"]["path_weights"] = args.codetr_weights

    if model_cfg['type'] == 'zero':
        from utils.inference.zero import ZeroAPI
        visual_queries = prepare_visual_queries(data_root, args.shot)
        text_queries = prepare_text_queries(data_root, args.shot)
        inferencer = ZeroAPI(**model_cfg['args'])
    elif model_cfg['type'] == 'codetr':
        from utils.inference.codetr import CoDETR
        inferencer = CoDETR(**model_cfg['args'])
    else:
        raise ValueError(f"Unsupported model type: {model_cfg['type']}")

    predictions = []
    for img_id in tqdm(coco.getImgIds()):
        img_dict = coco.loadImgs([img_id])[0]
        search_image = os.path.join(data_root, "test", img_dict["file_name"])

        if model_cfg['type'] == 'zero':
            if model_cfg['args']['prompt_type'] == 'visual':
                queries = visual_queries
            elif model_cfg['args']['prompt_type'] == 'text':
                queries = text_queries
            else:
                raise ValueError(
                    f"Unsupported prompt type: {model_cfg['args']['prompt_type']}"
                )

            inferencer.set_prompts(queries)
            result = inferencer.pred(search_image)
        elif model_cfg['type'] == 'codetr':
            result = inferencer.pred(search_image)
        else:
            raise ValueError(f"Unsupported model type: {model_cfg['type']}")

        for obj in result:
            x, y, w, h = obj["bbox_xywh"]
            predictions.append(
                {
                    "id": len(predictions) + 1,
                    "image_id": img_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "category_id": label2id[obj["class_name"]],
                    "iscrowd": 0,
                    "score": obj["score"],
                }
            )

    output_path = os.path.join(
        args.output,
        f"{args.data_name}_{args.shot}shot.json",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(predictions, handle)


if __name__ == "__main__":
    main()
