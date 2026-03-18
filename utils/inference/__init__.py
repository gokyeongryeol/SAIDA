from abc import ABC, abstractmethod
from typing import TypedDict


class TextPrompt(TypedDict):
    class_name: str


class VisualPrompt(TypedDict):
    class_name: str
    bbox_xywh: list[float]
    image_path: str


class DetectedObject(TypedDict):
    class_name: str
    bbox_xywh: list[float]
    score: float


class AbstractObjectDetector(ABC):
    @abstractmethod
    def pred(self, image_path: str)->list[DetectedObject]:
        pass

    @abstractmethod
    def get_class_names(self) -> list[str]:
        pass

    def set_text_prompts(self, text_prompts: list[TextPrompt]):
        raise NotImplementedError("This model does not support text prompts.")
    
    def set_visual_prompts(self, visual_prompts: list[VisualPrompt]):
        raise NotImplementedError("This model does not support visual prompts.")
