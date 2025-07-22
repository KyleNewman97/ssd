import pytest
import torch
from torch import Tensor
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from ssd.data import RemoveInvalidBoxesTransform
from ssd.structs import FrameLabels


class TestRemoveInvalidBoxesTransform:
    def test_init(self):
        """
        Test we can create a `RemoveInvalidBoxesTransform` instance.
        """
        transform = RemoveInvalidBoxesTransform()
        assert isinstance(transform, RemoveInvalidBoxesTransform)

    def test_forward(self):
        """
        Test that we remove boxes with a width or height of zero.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        image = torch.rand((3, 300, 300), dtype=dtype, device=device)
        boxes = BoundingBoxes(
            [[10, 10, 20, 20], [10, 10, 0, 20]],
            format=BoundingBoxFormat.CXCYWH,
            canvas_size=(300, 300),
            dtype=dtype,
            device=device,
        )  # type: ignore
        class_ids = torch.tensor([0, 1], dtype=torch.int, device=device)

        transform = RemoveInvalidBoxesTransform()
        out_image, out_objects = transform.forward(image, boxes, class_ids)

        # Check the output is correct
        assert isinstance(out_image, Tensor)
        assert out_image.allclose(image)
        assert isinstance(out_objects, FrameLabels)
        assert out_objects.boxes.shape[0] == 1
        assert out_objects.boxes.allclose(
            torch.tensor([[10, 10, 20, 20]], dtype=dtype, device=device)
        )
        assert out_objects.class_ids_with_background.shape[0] == 1
        assert out_objects.class_ids_with_background.equal(
            torch.tensor([0], dtype=torch.int, device=device)
        )

    def test_forward_invalid_box_format(self):
        """
        Test we raise an error when passing in an invalid box format.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        image = torch.rand((3, 300, 300), dtype=dtype, device=device)
        boxes = BoundingBoxes(
            [[10, 10, 20, 20]],
            format=BoundingBoxFormat.XYXY,
            canvas_size=(300, 300),
            dtype=dtype,
            device=device,
        )  # type: ignore
        class_ids = torch.tensor([0], dtype=torch.int, device=device)

        transform = RemoveInvalidBoxesTransform()

        with pytest.raises(ValueError):
            transform.forward(image, boxes, class_ids)
