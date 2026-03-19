import os
import io
import requests
import base64

from PIL import Image
from typing import Literal

from utils.inference import (
    AbstractObjectDetector,
    DetectedObject,
    TextPrompt,
    VisualPrompt,
)


class ZeroAPI(AbstractObjectDetector):
    def __init__(self, model_endpoint: str, prompt_type: str):
        self.model_endpoint = model_endpoint
        self.prompt_type = prompt_type

        self.id2label: list[str] = []
        self.label2id: dict[str, int] = {}

        self.visual_prompts: list[VisualPrompt] = []
        self.text_prompts: list[TextPrompt] = []

    def set_prompts(
        self, prompts: Literal[list[TextPrompt], list[VisualPrompt]],
    ):
        if self.prompt_type == "visual":
            self.visual_prompts = prompts
            self.text_prompts = []

        elif self.prompt_type == "text":
            self.visual_prompts = []
            self.text_prompts = prompts

        self.id2label = []
        for prompt in prompts:
            class_name = prompt["class_name"]
            if class_name not in self.id2label:
                self.id2label.append(class_name)

        self.label2id = {
            class_name: idx for idx, class_name in enumerate(self.id2label)
        }

    def _encode_image(self, image_path: str | Image.Image) -> str:
        if isinstance(image_path, str):
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            fext = os.path.splitext(image_path)[-1][1:].lower()  # get file extension without dot
        elif isinstance(image_path, Image.Image):
            buffered = io.BytesIO()
            image_path.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            fext = "png"
        else:
            raise NotImplementedError

        b64url_image = f'data:image/{fext};base64,' + base64.b64encode(image_bytes).decode("utf-8")
        return b64url_image

    def _bbox_xywh_to_xyxy(self, bbox_xywh: list[float]) -> list[float]:
        x, y, w, h = bbox_xywh
        return [x, y, x + w, y + h]

    def _bbox_xyxy_to_xywh(self, bbox_xyxy: list[float]) -> list[float]:
        x1, y1, x2, y2 = bbox_xyxy
        return [x1, y1, x2 - x1, y2 - y1]

    def pred(self, image_path: str) -> list[DetectedObject]:
        b64url_image = self._encode_image(image_path)

        payload = {
            'search_image': b64url_image,
            'apply_nms': 'batched_nms',
            'apply_topk': 300,
            'iou_threshold': 0.5,
            'queries': [],
        }

        if self.visual_prompts:
            prompts_by_image_path: dict[str, list[VisualPrompt]] = {}

            for prompt in self.visual_prompts:
                if prompt['image_path'] not in prompts_by_image_path:
                    prompts_by_image_path[prompt['image_path']] = []
                prompts_by_image_path[prompt['image_path']].append(prompt)

            for image_path, prompts in prompts_by_image_path.items():
                for prompt in prompts:
                    bbox_xyxy = self._bbox_xywh_to_xyxy(prompt['bbox_xywh'])
                    encoded_image = self._encode_image(image_path)

                    p = {
                        'prompt_image': encoded_image,
                        'prompts': [
                            {
                                'text': prompt['class_name'],
                                'box': bbox_xyxy,
                                'box_threshold': 0.03,
                                'multimodal_threshold': 0.03,
                            }
                        ]
                    }
                    payload['queries'].append(p)
        elif self.text_prompts:
            for prompt in self.text_prompts:
                p = {
                    'propmt_image': '',
                    'prompts': [
                        {
                            'text': prompt['class_name'],
                            'box': [],
                            'box_threshold': 0.03,
                            'multimodal_threshold': 0.03,
                        }
                    ]
                }
                payload['queries'].append(p)
        else:
            raise ValueError(
                "No prompts set for the model."
                "Please set either visual prompts or text prompts before calling pred()."
            )

        resp = requests.post(self.model_endpoint, json=payload)
        resp.raise_for_status()
        response_data = resp.json()

        predictions = response_data['output'][0]
        n_boxes = len(predictions['text'])

        result: list[DetectedObject] = []
        for boxidx in range(n_boxes):
            class_name = predictions['text'][boxidx]
            e = DetectedObject(
                class_name=class_name,
                bbox_xywh=self._bbox_xyxy_to_xywh(predictions['boxes'][boxidx]),
                score=predictions['scores'][boxidx],
            )
            result.append(e)
        return result

    def get_class_names(self) -> list[str]:
        return self.id2label
