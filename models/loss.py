from torch import Tensor
from torch.nn import Module, BCELoss, MultiLabelMarginLoss, MSELoss

from common import TensorData


class MTLoss(Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn_bin = BCELoss()
        self.loss_fn_score = MSELoss()
        self.loss_fn_attribute = MultiLabelMarginLoss()

    def forward(self, input_tensor: TensorData, label_tensor: TensorData) -> Tensor:
        loss_bin: Tensor = self.loss_fn_bin(input_tensor["binary"], label_tensor["binary"])
        loss_score: Tensor = self.loss_fn_score(input_tensor["score"], label_tensor["score"])
        loss_attribute: Tensor = self.loss_fn_attribute(input_tensor["attribute"], label_tensor["attribute"])

        # 这里将三个子任务的损失值相加
        return loss_bin + loss_score + loss_attribute
