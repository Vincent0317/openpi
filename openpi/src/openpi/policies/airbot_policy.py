import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_airbot_example() -> dict:
    # 创建Airbot数据示例
    return {
        "state": np.random.rand(7),  # 7维状态空间 (6个关节 + 1个夹持器)
        "image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),  # 顶视图
        "wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),  # 手腕视图
        "task": "do something",  # 任务描述
    }


def _parse_image(image) -> np.ndarray:
    # 图像格式转换
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AirbotInputs(transforms.DataTransformFn):

    # 将输入数据转换为模型期望的格式。适用于训练和推理。适配Airbot的输入格式。
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        
        # 处理机器人状态 - 将7维状态扩展到32维 (pi0 base必须要补0)
        state = data["state"]
        if len(state) < self.action_dim:
            # 用零填充扩展到32维
            padded_state = np.zeros(self.action_dim)
            padded_state[:len(state)] = state
            state = padded_state
        
        # 对pi0模型进行填充掩码
        mask_padding = self.model_type == _model.ModelType.PI0

        # 处理机器人状态
        # Airbot: 7维状态 (6个关节 + 1个夹持器)
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # 处理图像输入
        # Airbot有两个视角:顶视图和手腕视图
        base_image = _parse_image(data["top_image"])  # 顶视图
        wrist_image = _parse_image(data["wrist_image"])  # 手腕视图

        # 创建输入字典
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Airbot只有一个手腕摄像头,不存在的数据用零填充
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # 动作填充(仅用于训练)
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # 添加任务提示词
        if "task" in data:
            inputs["prompt"] = data["task"]
        elif "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AirbotOutputs(transforms.DataTransformFn):

    # 将模型输出转换回数据集特定格式。仅用于推理。适配Airbot的输出格式。

    def __call__(self, data: dict) -> dict:
        
        return {"actions": np.asarray(data["actions"][:5, :7])}