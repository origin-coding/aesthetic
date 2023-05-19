from typing import TypedDict, Tuple, List

from torch import Tensor


# 训练过程中用到的变量类型
class TensorData(TypedDict):
    binary: Tensor
    attribute: Tensor
    score: Tensor


class TrainData(TypedDict):
    input_tensor: TensorData
    label_tensor: TensorData


StepOutput = Tuple[TensorData, TensorData]


# 应用程序用到的数据类型
class AttributeResult(TypedDict):
    """
    图像美学多标签评价的标签名
    """
    balancing_element: bool  # 平衡
    content: bool  # 内容
    color_harmony: bool  # 颜色和谐
    depth_of_field: bool  # 景深
    lighting: bool  # 光照
    motion_blur: bool  # 运动模糊
    object_emphasis: bool  # 主体强调
    rule_of_thirds: bool  # 三分法
    vivid_color: bool  # 颜色鲜明
    repetition: bool  # 重复
    symmetry: bool  # 对称性


def new_attribute_result(results: List[bool]) -> AttributeResult:
    assert len(results) == 11  # 首先判断列表长度是否正确
    return AttributeResult(
        depth_of_field=results[0],
        balancing_element=results[1],
        color_harmony=results[2],
        content=results[3],
        lighting=results[4],
        motion_blur=results[5],
        object_emphasis=results[6],
        repetition=results[7],
        symmetry=results[8],
        rule_of_thirds=results[9],
        vivid_color=results[10]
    )


class AssessResult(TypedDict):
    binary: bool
    score: float
    attribute: AttributeResult
