import torch
from torch import nn, Tensor
from torchvision.transforms.functional import resize


class LetterboxTransform(nn.Module):
    def __init__(
        self, width: int = 300, height: int = 300, dtype: torch.dtype = torch.float32
    ):
        self.desired_width = width
        self.desired_height = height
        self.dtype = dtype

    def __call__(
        self, image: Tensor, objects: Tensor, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """
        Applies the letterbox transform to both the image and the objects.

        Parameters
        ----------
        image:
            A single image tensor with dimensions of `(channels, height, width)`.

        objects:
            The classification and bounding box of objects within the image. This is
            structured as `(num_objects, 5)`. With the last dimension containing the
            following: `(class_id, cx, cy, w, h)`. The bounding box is defined in
            normalised space (betweem 0 and 1).

        device:
            The device to put place the data onto.

        Returns
        -------
        output_image:
            A letterboxed version of the original input image. This will have a shape of
            `(channels, desired_height, desired_width)`.

        objects:
            A letterboxed version of the original objects. This will have the same shape
            as the input objects: `(num_objects, 5)`.
        """
        desired_wh_ratio = self.desired_width / self.desired_height
        image_wh_ratio = image.shape[2] / image.shape[1]

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
        resized_image = resize(image, [new_height, new_width])

        # Determine x and y start and ends
        x_start = (self.desired_width - new_width) // 2
        x_end = x_start + new_width
        y_start = (self.desired_height - new_height) // 2
        y_end = y_start + new_height

        # Create the output image
        desired_shape = (image.shape[0], self.desired_height, self.desired_width)
        output_image = torch.zeros(desired_shape, dtype=self.dtype, device=device)
        output_image[:, y_start:y_end, x_start:x_end] = resized_image

        # Adjust the object positions
        out_objects = objects.clone()
        if desired_wh_ratio <= image_wh_ratio:
            # When bound by the width adjust the y values
            out_objects[:, 2::2] *= new_height / self.desired_height
            out_objects[:, 2] += y_start / self.desired_height
        else:
            # When bound by the height adjust the x values
            out_objects[:, 1::2] *= new_width / self.desired_width
            out_objects[:, 1] += x_start / self.desired_width

        return output_image, out_objects
