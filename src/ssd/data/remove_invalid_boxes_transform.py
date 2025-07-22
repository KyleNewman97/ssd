from torch import nn, Tensor
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from ssd.structs import FrameLabels


class RemoveInvalidBoxesTransform(nn.Module):
    """
    Removes bounding boxes with zero width or height.
    """

    def forward(
        self, image: Tensor, boxes: BoundingBoxes, class_ids_with_background: Tensor
    ) -> tuple[Tensor, FrameLabels]:
        if boxes.format != BoundingBoxFormat.CXCYWH:
            raise ValueError(f"Expected box format of cxcywh but got {boxes.format}.")

        # Remove boxes with zero width or height
        cleaned_boxes = boxes.clone()
        mask = (cleaned_boxes[:, 2] > 1).bitwise_and(cleaned_boxes[:, 3] > 1)
        cleaned_boxes = cleaned_boxes[mask, :]
        cleaned_class_ids = class_ids_with_background[mask]

        labels = FrameLabels(
            boxes=cleaned_boxes,
            class_ids_with_background=cleaned_class_ids,
        )

        return image, labels
