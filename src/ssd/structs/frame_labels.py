from pydantic import BaseModel, ConfigDict
from torch import Tensor


class FrameLabels(BaseModel):
    boxes: Tensor
    labels: Tensor

    model_config = ConfigDict(arbitrary_types_allowed=True)
