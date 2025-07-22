from torch import nn, Tensor
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat


class RemoveInvalidBoxesTransform(nn.Module):
    """
    Removes bounding boxes with zero width or height.
    """

    def forward(
        self, image: Tensor, boxes: BoundingBoxes
    ) -> tuple[Tensor, BoundingBoxes]:
        if boxes.format != BoundingBoxFormat.CXCYWH:
            raise ValueError(f"Expected box format of cxcywh but got {boxes.format}.")

        # Remove boxes with zero width or height
        cleaned_boxes = boxes.clone()
        mask = (cleaned_boxes[:, 2] > 1).bitwise_and(cleaned_boxes[:, 3] > 1)
        cleaned_boxes = cleaned_boxes[mask, :]

        out_boxes = BoundingBoxes(
            cleaned_boxes, format=boxes.format, canvas_size=boxes.canvas_size
        )  # type: ignore

        return image, out_boxes
