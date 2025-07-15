import torch
from pydantic import BaseModel
from torch import nn, Tensor
from torchvision.transforms.functional import resize

from ssd.structs import FrameLabels


class LetterboxTransform(nn.Module):
    def __init__(
        self, width: int = 300, height: int = 300, dtype: torch.dtype = torch.float32
    ):
        self.desired_width = width
        self.desired_height = height
        self.dtype = dtype

    def _calculate_transform_params(self, image_width: int, image_height: int):
        desired_wh_ratio = self.desired_width / self.desired_height
        image_wh_ratio = image_width / image_height

        # Find the new width and height of the image after resizing it to fit into the
        # desired width and height - without loosing the aspect ratio
        if desired_wh_ratio <= image_wh_ratio:
            # Find the dimensions when we are bound by the width
            new_width = self.desired_width
            new_height = int(new_width / image_wh_ratio)
        else:
            # Find the dimensions when we are bound by the height
            new_height = self.desired_height
            new_width = int(new_height * image_wh_ratio)

        # Determine x and y start and ends
        x_start = (self.desired_width - new_width) // 2
        x_end = x_start + new_width
        y_start = (self.desired_height - new_height) // 2
        y_end = y_start + new_height

        return TransformParams(
            desired_wh_ratio=desired_wh_ratio,
            image_wh_ratio=image_wh_ratio,
            new_width=new_width,
            new_height=new_height,
            x_start=x_start,
            x_end=x_end,
            y_start=y_start,
            y_end=y_end,
        )

    def transform_image(self, image: Tensor, device: torch.device) -> Tensor:
        params = self._calculate_transform_params(image.shape[2], image.shape[1])

        # Create the output image
        resized_image = resize(image, [params.new_height, params.new_width])
        desired_shape = (image.shape[0], self.desired_height, self.desired_width)
        output_image = torch.zeros(desired_shape, dtype=self.dtype, device=device)
        output_image[
            :, params.y_start : params.y_end, params.x_start : params.x_end
        ] = resized_image

        return output_image

    def transform_objects(
        self, objects: FrameLabels, image_width: int, image_height: int
    ) -> FrameLabels:
        params = self._calculate_transform_params(image_width, image_height)

        # Adjust the object positions
        out_boxes = objects.boxes.clone()
        if params.desired_wh_ratio <= params.image_wh_ratio:
            # When bound by the width adjust the y values
            out_boxes[:, 1::2] *= params.new_height / self.desired_height
            out_boxes[:, 1] += params.y_start / self.desired_height
        else:
            # When bound by the height adjust the x values
            out_boxes[:, 0::2] *= params.new_width / self.desired_width
            out_boxes[:, 0] += params.y_end / self.desired_width

        return FrameLabels(boxes=out_boxes, class_ids=objects.class_ids)

    def __call__(
        self, image: Tensor, objects: FrameLabels, device: torch.device
    ) -> tuple[Tensor, FrameLabels]:
        """
        Applies the letterbox transform to both the image and the objects.

        Parameters
        ----------
        image:
            A single image tensor with dimensions of `(channels, height, width)`.

        objects:
            Labelled objects for the image.

        device:
            The device to put place the data onto.

        Returns
        -------
        output_image:
            A letterboxed version of the original input image. This will have a shape of
            `(channels, desired_height, desired_width)`.

        out_objects:
            A letterboxed version of the image's objects.
        """

        letterbox_image = self.transform_image(image, device)
        letterbox_objs = self.transform_objects(objects, image.shape[2], image.shape[1])

        return letterbox_image, letterbox_objs


class TransformParams(BaseModel):
    desired_wh_ratio: float
    image_wh_ratio: float
    new_width: int
    new_height: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
