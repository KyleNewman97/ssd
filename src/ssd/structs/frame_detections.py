from pydantic import BaseModel, ConfigDict
from torch import Tensor


class FrameDetections(BaseModel):
    boxes: Tensor
    scores: Tensor
    labels: Tensor

    model_config = ConfigDict(arbitrary_types_allowed=True)
