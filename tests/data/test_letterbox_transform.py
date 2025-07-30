from pathlib import Path

import cv2
import pytest
import torch
from torch import Tensor

from ssd.data import LetterboxTransform, SSDDataset


class TestLetterboxTransform:
    @pytest.fixture(autouse=True)
    def transform(self) -> LetterboxTransform:
        return LetterboxTransform(width=300, height=300, dtype=torch.float32)

    def test_init(self, transform: LetterboxTransform):
        """
        Test we can initialise a LetterboxTransform object.
        """
        assert isinstance(transform, LetterboxTransform)

    def test_call_landscape(self, transform: LetterboxTransform):
        """
        Test that we can apply the letterbox transform to an image and its label.
        """
        device = torch.device("cpu")
        dtype = torch.float32

        # Read in a test image
        np_image = cv2.imread("test_data/dog.jpg")
        image = torch.tensor(np_image, dtype=dtype, device=device)
        image = image.permute((2, 0, 1))

        # Define labels
        objects = SSDDataset.read_label_file(Path("test_data/dog.txt"), device, dtype)

        # Apply the letterbox transform
        out_image, out_objects = transform(image, objects, device)

        # Ensure output image type and shape is correct
        assert isinstance(out_image, Tensor)
        assert out_image.shape == (
            image.shape[0],
            transform.desired_height,
            transform.desired_width,
        )
        assert out_image.device == device
        assert out_image.dtype == dtype

        # Ensure the output image is correct
        out_image = out_image.permute((1, 2, 0)).to(dtype=torch.uint8)
        expected_image = cv2.imread("test_data/letterboxed_dog.png")
        expected_image = torch.tensor(expected_image, dtype=torch.uint8)
        assert out_image.shape == expected_image.shape
        assert out_image.equal(expected_image)

        # Ensure the adjusted objects are correct
        expected_class_ids = torch.tensor([1], dtype=torch.int, device=device)
        expected_boxes = torch.tensor(
            [[0.5358, 0.4950, 0.3483, 0.4038]], dtype=dtype, device=device
        )
        assert out_objects.class_ids_with_background.device == device
        assert out_objects.boxes.device == device
        assert out_objects.boxes.dtype == dtype
        assert out_objects.class_ids_with_background.equal(expected_class_ids)
        assert out_objects.boxes.allclose(expected_boxes, rtol=0.001)

    def test_call_portrait(self, transform: LetterboxTransform):
        """
        Test we can perform the letterbox operation correctly on portrait images.
        """
        device = torch.device("cpu")
        dtype = torch.float32

        # Read in a test image
        np_image = cv2.imread("test_data/portrait.jpg")
        image = torch.tensor(np_image, dtype=dtype, device=device)
        image = image.permute((2, 0, 1))

        # Define labels
        objects = SSDDataset.read_label_file(
            Path("test_data/portrait.txt"), device, dtype
        )

        # Apply the letterbox transform
        out_image, out_objects = transform(image, objects, device)

        # Ensure output image type and shape is correct
        assert isinstance(out_image, Tensor)
        assert out_image.shape == (
            image.shape[0],
            transform.desired_height,
            transform.desired_width,
        )
        assert out_image.device == device
        assert out_image.dtype == dtype

        # Ensure the output image is correct
        out_image = out_image.permute((1, 2, 0)).to(dtype=torch.uint8)
        expected_image = cv2.imread("test_data/letterboxed_portrait.png")
        expected_image = torch.tensor(expected_image, dtype=torch.uint8)
        assert out_image.shape == expected_image.shape
        assert out_image.equal(expected_image)

        # Ensure the adjusted objects are correct
        expected_boxes = torch.tensor(
            [
                [0.4545, 0.3199, 0.5378, 0.3004],
                [0.5067, 0.5183, 0.7331, 0.4560],
                [0.4281, 0.2237, 0.4401, 0.2195],
                [0.5082, 0.6877, 0.7303, 0.6000],
            ],
            dtype=dtype,
            device=device,
        )
        assert out_objects.class_ids_with_background.device == device
        assert out_objects.boxes.device == device
        assert out_objects.boxes.dtype == dtype
        assert out_objects.class_ids_with_background.equal(
            objects.class_ids_with_background
        )
        assert out_objects.boxes.allclose(expected_boxes, rtol=0.001)
