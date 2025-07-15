from pydantic import BaseModel, ConfigDict
from torch import Tensor


class FrameDetections(BaseModel):
    boxes: Tensor
    scores: Tensor
    class_ids: Tensor

    model_config = ConfigDict(arbitrary_types_allowed=True)
