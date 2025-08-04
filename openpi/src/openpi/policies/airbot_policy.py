import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_airbot_example() -> dict:
    # ����Airbot����ʾ��
    return {
        "state": np.random.rand(7),  # 7ά״̬�ռ� (6���ؽ� + 1���г���)
        "image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),  # ����ͼ
        "wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),  # ������ͼ
        "task": "do something",  # ��������
    }


def _parse_image(image) -> np.ndarray:
    # ͼ���ʽת��
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AirbotInputs(transforms.DataTransformFn):

    # ����������ת��Ϊģ�������ĸ�ʽ��������ѵ������������Airbot�������ʽ��
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        
        # ���������״̬ - ��7ά״̬��չ��32ά (pi0 base����Ҫ��0)
        state = data["state"]
        if len(state) < self.action_dim:
            # ���������չ��32ά
            padded_state = np.zeros(self.action_dim)
            padded_state[:len(state)] = state
            state = padded_state
        
        # ��pi0ģ�ͽ����������
        mask_padding = self.model_type == _model.ModelType.PI0

        # ���������״̬
        # Airbot: 7ά״̬ (6���ؽ� + 1���г���)
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # ����ͼ������
        # Airbot�������ӽ�:����ͼ��������ͼ
        base_image = _parse_image(data["top_image"])  # ����ͼ
        wrist_image = _parse_image(data["wrist_image"])  # ������ͼ

        # ���������ֵ�
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Airbotֻ��һ����������ͷ,�����ڵ������������
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # �������(������ѵ��)
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # ���������ʾ��
        if "task" in data:
            inputs["prompt"] = data["task"]
        elif "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AirbotOutputs(transforms.DataTransformFn):

    # ��ģ�����ת�������ݼ��ض���ʽ����������������Airbot�������ʽ��

    def __call__(self, data: dict) -> dict:
        
        return {"actions": np.asarray(data["actions"][:5, :7])}