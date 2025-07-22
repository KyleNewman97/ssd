from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat


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

    def tv_boxes(self, image_width: int, image_height: int) -> BoundingBoxes:
        """
        Return the bounding boxes as an `tv_tensors.BoundingBoxes`. This format is
        useful when using torchvision transforms.

        Parameters
        ----------
        image_width:
            The width of the image in pixels.

        image_height:
            The height of the image in pixels.

        Returns
        -------
        boxes:
            tv_tensors.BoundingBoxes
        """
        # Convert normalised box coords to image coords
        boxes = self.boxes.clone()
        boxes[:, ::2] *= image_width
        boxes[:, 1::2] *= image_height

        return BoundingBoxes(
            boxes,
            format=BoundingBoxFormat.CXCYWH,
            canvas_size=(image_height, image_width),
        )  # type: ignore

    model_config = ConfigDict(arbitrary_types_allowed=True)
