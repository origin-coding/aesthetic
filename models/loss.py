import math
from typing import Optional, Tuple

from torch import Tensor
from torch.nn import Module, BCEWithLogitsLoss, MultiLabelMarginLoss, MSELoss

from common import TensorData


class MTLoss(Module):

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn_bin = BCEWithLogitsLoss()
        self.loss_fn_score = MSELoss()
        self.loss_fn_attribute = MultiLabelMarginLoss()

    def forward(self, output_tensor: TensorData, label_tensor: TensorData) -> Tensor:
        # noinspection DuplicatedCode
        loss_bin: Tensor = self.loss_fn_bin(output_tensor["binary"].float(), label_tensor["binary"].float())
        loss_score: Tensor = self.loss_fn_score(output_tensor["score"].float(), label_tensor["score"].float())
        loss_attribute: Tensor = self.loss_fn_attribute(output_tensor["attribute"].float(),
                                                        label_tensor["attribute"].long())

        # 这里将三个子任务的损失值相加
        return loss_bin + loss_score + loss_attribute


losses = Optional[Tuple[Tensor, Tensor, Tensor]]


class MTDwa(Module):
    def __init__(self):
        super().__init__()
        self.loss_fn_bin = BCEWithLogitsLoss()
        self.loss_fn_score = MSELoss()
        self.loss_fn_attribute = MultiLabelMarginLoss()

        self.loss_t1: losses = None
        self.loss_t2: losses = None

    def forward(self, output_tensor: TensorData, label_tensor: TensorData) -> Tensor:
        # noinspection DuplicatedCode
        loss_bin: Tensor = self.loss_fn_bin(output_tensor["binary"].float(), label_tensor["binary"].float())
        loss_score: Tensor = self.loss_fn_score(output_tensor["score"].float(), label_tensor["score"].float())
        loss_attribute: Tensor = self.loss_fn_attribute(output_tensor["attribute"].float(),
                                                        label_tensor["attribute"].long())

        weights = MTDwa.dynamic_weight_averaging(self.loss_t1, self.loss_t2)

        results = loss_bin * weights[0], loss_score * weights[1], loss_attribute * weights[2]

        self.loss_t2 = self.loss_t1
        self.loss_t1 = results

        # 这里将三个子任务的损失值相加
        return sum(results)

    @staticmethod
    def dynamic_weight_averaging(loss_t1: losses, loss_t2: losses) -> Tuple[float, float, float]:
        # 如果是第一轮或第二轮，那么三个子任务权重相同
        if not loss_t1 or not loss_t2:
            return 1.0, 1.0, 1.0

        loss_t1 = (i.detach().item() for i in loss_t1)
        loss_t2 = (i.detach().item() for i in loss_t2)

        weight = [l1 / l2 for l1, l2 in zip(loss_t1, loss_t2)]
        lamb = [math.exp(v / 10) for v in weight]
        lamb_sum = sum(lamb)

        return 3 * lamb[0] / lamb_sum, 3 * lamb[1] / lamb_sum, 3 * lamb[2] / lamb_sum
