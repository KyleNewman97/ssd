from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor


class FrameLabels(BaseModel):
    """
    Contains the ground truth objects found in an image.
    """

    boxes: Tensor = Field(
        description=(
            "The bounding boxes of objects. This will have a shape of `(num_objects, 4)"
            "`, with the last dimension being structured as `(cx, cy, w, h)`. All "
            "values are normalised between 0 and 1."
        )
    )
    class_ids_with_background: Tensor = Field(
        description=(
            "The class IDs of objects. This will have a shape of `(num_objects,)`."
        )
    )

    @property
    def class_ids(self) -> Tensor:
        """
        The class IDs with no background class.
        """
        return self.class_ids_with_background - 1

    model_config = ConfigDict(arbitrary_types_allowed=True)
