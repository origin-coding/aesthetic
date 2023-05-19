from pathlib import Path
from urllib.parse import urlparse

import torch
from PIL import Image
from PySide6.QtCore import QObject, Slot, Signal
from torch import Tensor

from common import image_transforms, TensorData, AssessResult, new_attribute_result
from models import MTAesthetic


class Context(QObject):
    def __init__(self):
        super().__init__()
        # 设置默认参数：使用attention，kernel size为5，不使用DWA
        self.use_attention = True
        self.kernel_size = 5
        self.use_dwa = False
        # 加载模型
        self.model = MTAesthetic(channels=1024, kernel_size=self.kernel_size, use_attention=self.use_attention)

    @Slot(str)
    def assess_image(self, image_url: str) -> None:
        image_path = urlparse(image_url, scheme="file").path.removeprefix("/")
        image_path = Path(image_path).absolute()

        # 如果图片不存在，直接返回（虽然可能性不大）
        if not image_path.exists():
            return

        image = Image.open(image_path)
        image_tensor: Tensor = image_transforms(image).unsqueeze(0)
        input_tensor: TensorData = TensorData(binary=image_tensor, attribute=image_tensor, score=image_tensor)
        output_tensor: TensorData = self.model(input_tensor)

        # 将输出转换成对应的类型，并传递给页面
        assess_result = Context.process_output(output_tensor)
        self.send_result(assess_result)

    def send_result(self, assess_result: AssessResult):
        self.setBinary.emit(assess_result["binary"])
        self.setScore.emit(assess_result["score"])
        self.setBalancingElement.emit(assess_result["attribute"]["balancing_element"])
        self.setContent.emit(assess_result["attribute"]["content"])
        self.setColorHarmony.emit(assess_result["attribute"]["color_harmony"])
        self.setDepthOfField.emit(assess_result["attribute"]["depth_of_field"])
        self.setLighting.emit(assess_result["attribute"]["lighting"])
        self.setMotionBlur.emit(assess_result["attribute"]["motion_blur"])
        self.setObjectEmphasis.emit(assess_result["attribute"]["object_emphasis"])
        self.setRuleOfThirds.emit(assess_result["attribute"]["rule_of_thirds"])
        self.setVividColor.emit(assess_result["attribute"]["vivid_color"])
        self.setRepetition.emit(assess_result["attribute"]["repetition"])
        self.setSymmetry.emit(assess_result["attribute"]["symmetry"])

    @staticmethod
    def process_output(output_tensor: TensorData) -> AssessResult:
        result_binary: bool = output_tensor["binary"].item() > 0.5

        scores = torch.tensor([i for i in range(1, 11)], dtype=torch.float)
        result_score: Tensor = torch.dot(output_tensor["score"].squeeze(0), scores)
        result_score: float = round(result_score.item(), 3)

        result_attribute: Tensor = output_tensor["attribute"].squeeze(0) > 0.5
        result_attribute: list = result_attribute.tolist()

        return AssessResult(
            binary=result_binary,
            score=result_score,
            attribute=new_attribute_result(result_attribute)
        )

    # Signals
    setBinary = Signal(bool)
    setScore = Signal(float)

    setBalancingElement = Signal(bool)
    setContent = Signal(bool)
    setColorHarmony = Signal(bool)
    setDepthOfField = Signal(bool)
    setLighting = Signal(bool)
    setMotionBlur = Signal(bool)
    setObjectEmphasis = Signal(bool)
    setRuleOfThirds = Signal(bool)
    setVividColor = Signal(bool)
    setRepetition = Signal(bool)
    setSymmetry = Signal(bool)
