from torch import Tensor
from torch.nn import Module, BCEWithLogitsLoss, MultiLabelMarginLoss, MSELoss

from common import TensorData


class MTLoss(Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn_bin = BCEWithLogitsLoss()
        self.loss_fn_score = MSELoss()
        self.loss_fn_attribute = MultiLabelMarginLoss()

    def forward(self, output_tensor: TensorData, label_tensor: TensorData) -> Tensor:
        loss_bin: Tensor = self.loss_fn_bin(output_tensor["binary"].float(), label_tensor["binary"].float())
        loss_score: Tensor = self.loss_fn_score(output_tensor["score"].float(), label_tensor["score"].float())
        loss_attribute: Tensor = self.loss_fn_attribute(output_tensor["attribute"].float(),
                                                        label_tensor["attribute"].long())

        # 这里将三个子任务的损失值相加
        return loss_bin + loss_score + loss_attribute
