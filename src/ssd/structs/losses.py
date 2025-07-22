from pydantic import BaseModel, ConfigDict
from torch import Tensor


class Losses(BaseModel):
    box_loss: Tensor
    class_loss: Tensor

    model_config = ConfigDict(arbitrary_types_allowed=True)
