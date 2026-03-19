import copy

from utils.inference import (
    AbstractObjectDetector,
    DetectedObject,
    TextPrompt,
    VisualPrompt,
)

from mmdet.apis import DetInferencer
from mmengine.config import Config, ConfigDict


class CoDETR(AbstractObjectDetector):
    def __init__(self, path_cfg: str, path_weights: str, class_names: list[str], use_tta=False):
        cfg = Config.fromfile(path_cfg)

        if use_tta:
            cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
            cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

        self.inferencer = DetInferencer(
            model=cfg, weights=path_weights, show_progress=False,
        )
        self.clsas_names = class_names

    def set_visual_prompts(self, visual_prompts: list[VisualPrompt]):
        raise NotImplementedError("CODETR does not support visual prompts.")

    def set_text_prompts(self, text_prompts: list[TextPrompt]):
        raise NotImplementedError("CODETR does not support text prompts.")

    def pred(self, image_path: str) -> list[DetectedObject]:
        output = self.inferencer(image_path, draw_pred=False)['predictions'][0]

        result = []
        if "bboxes" in output and len(output["bboxes"]) > 0:
            for box, score, c_id in zip(output["bboxes"], output["scores"], output["labels"]):
                x1, y1, x2, y2 = box
                result.append(
                    {
                        "class_name": self.clsas_names[c_id],
                        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

        return result

    def get_class_names(self) -> list[str]:
        return self.clsas_names
