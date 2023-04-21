from typing import Tuple

import torch
from ignite.metrics import Loss, Accuracy
from torch import Tensor
from torch.nn import MultiLabelMarginLoss, BCEWithLogitsLoss, MSELoss

from common import StepOutput
from models import MTLoss, MTDwa
from .config import EngineMetrics, Configuration


def extract_binary(output: StepOutput) -> Tuple[Tensor, Tensor]:
    """
    辅助函数，用于提取美学质量高低的二分类问题数据
    :param output: 每个step的输出内容
    :return: 一个元组，包含抽取后的output_tensor(y)和label_tensor(y_pred)
    """
    return output[0]["binary"].float(), output[1]["binary"].float()


def extract_binary_accuracy(output: StepOutput) -> Tuple[Tensor, Tensor]:
    """
    辅助函数，用于提取美学质量高低的二分类问题数据，这里第一个参数需要取整，用于计算准确率
    :param output: 每个step的输出内容
    :return: 一个元组，包含抽取后的output_tensor(y)和label_tensor(y_pred)
    """
    return output[0]["binary"].round().float(), output[1]["binary"].float()


def extract_score(output: StepOutput) -> Tuple[Tensor, Tensor]:
    """
    辅助函数，用于提取美学质量评分的回归问题数据
    :param output: 每个step的输出内容
    :return: 一个元组，包含抽取后的output_tensor(y)和label_tensor(y_pred)
    """
    return output[0]["score"].float(), output[1]["score"].float()


def extract_attribute(output: StepOutput) -> Tuple[Tensor, Tensor]:
    """
    辅助函数，用于提取美学质量具体类别的多标签分类问题数据
    :param output: 每个step的输出内容
    :return: 一个元组，包含抽取后的output_tensor(y)和label_tensor(y_pred)
    """
    return output[0]["attribute"].float(), output[1]["attribute"].long()


def setup_metrics(device: torch.device, config: Configuration) -> EngineMetrics:
    """
    用于创建训练过程中需要用到的metrics，包括总体Loss、三个子任务各自的Loss和二分类问题的准确率
    :param device: 数据所在的device，一般是GPU
    :param config: 配置项，用到了use_dwa
    :return: 一个字典，key是每个metric的名称，value是对应的metric
    """
    return {
        "loss": Loss(loss_fn=MTLoss() if not config.use_dwa else MTDwa(), device=device),
        "bin_loss": Loss(loss_fn=BCEWithLogitsLoss(), output_transform=extract_binary, device=device),
        "score_loss": Loss(loss_fn=MSELoss(), output_transform=extract_score, device=device),
        "attr_loss": Loss(loss_fn=MultiLabelMarginLoss(), output_transform=extract_attribute, device=device),
        # Accuracy metric要求output_tensor(y)只能是0或1
        "bin_acc": Accuracy(output_transform=extract_binary_accuracy, device=device),
    }
