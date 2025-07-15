from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
import torch
import numpy as np
from PIL import Image
from torch import Tensor

from ssd.data import LetterboxTransform, SSDDataset
from ssd.structs import FrameLabels


class TestSSDDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.images_path = self.temp_path / "images"
        self.labels_path = self.temp_path / "labels"

        self.images_path.mkdir(exist_ok=True, parents=True)
        self.labels_path.mkdir(exist_ok=True, parents=True)

        yield

        self.temp_dir.cleanup()

    @pytest.fixture(autouse=True)
    def device(self) -> torch.device:
        return torch.device("cpu")

    def test_init_valid_dataset(self, device: torch.device):
        """
        Test that we keep all valid samples when initialising the dataset.
        """

        # Make a single image file and label file that are valid
        uuid = f"{uuid4()}"

        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")

        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("0 0.1 0.1 0.1 0.1")

        # Try to initialise the dataset
        transform = LetterboxTransform()
        dataset = SSDDataset(
            self.images_path, self.labels_path, 2, transform, device, torch.float32
        )
        assert isinstance(dataset, SSDDataset)
        assert len(dataset) == 1

    def test_init_missing_label(self, device: torch.device):
        """
        Test that we remove samples when we are missing the label.
        """

        # Make a single image file
        uuid = f"{uuid4()}"

        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")

        # Try to initialise the dataset
        transform = LetterboxTransform()
        dataset = SSDDataset(
            self.images_path, self.labels_path, 2, transform, device, torch.float32
        )
        assert isinstance(dataset, SSDDataset)
        assert len(dataset) == 0

    def test_init_invalid_label_num_elements(self, device: torch.device):
        """
        Test that we remove samples that contain label rows with not enough elements.
        """

        # Make a single image file
        uuid = f"{uuid4()}"

        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")

        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("0 0.1 0.1 0.1")

        # Try to initialise the dataset
        transform = LetterboxTransform()
        dataset = SSDDataset(
            self.images_path, self.labels_path, 2, transform, device, torch.float32
        )
        assert isinstance(dataset, SSDDataset)
        assert len(dataset) == 0

    def test_init_invalid_label_class(self, device: torch.device):
        """
        Test that we remove samples that contain an invalid class ID.
        """

        # Make a single image file
        uuid = f"{uuid4()}"

        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")

        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("2 0.1 0.1 0.1 0.1")

        # Try to initialise the dataset
        transform = LetterboxTransform()
        dataset = SSDDataset(
            self.images_path, self.labels_path, 2, transform, device, torch.float32
        )
        assert isinstance(dataset, SSDDataset)
        assert len(dataset) == 0

    def test_init_invalid_label_box(self, device: torch.device):
        """
        Test that we remove samples that contain an invalid box.
        """

        # Make a single image file
        uuid = f"{uuid4()}"

        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")

        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("0 0.1 0.1 0.1 -0.1")

        # Try to initialise the dataset
        transform = LetterboxTransform()
        dataset = SSDDataset(
            self.images_path, self.labels_path, 2, transform, device, torch.float32
        )
        assert isinstance(dataset, SSDDataset)
        assert len(dataset) == 0

    def test_read_label_file(self, device: torch.device):
        """
        Test that we can read in a valid label file.
        """
        with TemporaryDirectory() as temp_dir:
            # Create a dummy label file
            file = Path(temp_dir) / "label.txt"
            with open(file, "w") as fp:
                fp.write("0 0.5358 0.4942 0.3483 0.7085\n1 0.2 0.4 0.02 0.08\n")

            # Try to read in the labels
            objects = SSDDataset.read_label_file(file, device, torch.float32)

        # Check the contents is correct
        assert objects.boxes.device == device
        assert objects.class_ids.device == device
        assert objects.boxes.shape == (2, 4)
        assert objects.class_ids.shape == (2,)

    def test_get_item(self, device: torch.device):
        """
        Test that we can get an item from the dataset.
        """
        dtype = torch.float32

        # Make a single image file and label file that are valid
        uuid = f"{uuid4()}"

        im_data = np.random.randint(0, 255, (320, 320, 3)).astype(np.uint8)
        im = Image.fromarray(im_data)
        im.save(self.images_path / f"{uuid}.png")

        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("0 0.1 0.1 0.1 0.1")

        # Try to get a sample from the dataset
        num_classes = 2
        height, width = 256, 512
        transform = LetterboxTransform(height=height, width=width, dtype=dtype)
        dataset = SSDDataset(
            self.images_path,
            self.labels_path,
            num_classes,
            transform,
            device,
            torch.float32,
        )
        image, objects = dataset[0]

        # Check the output shapes are correct
        assert isinstance(image, Tensor)
        assert image.device == device
        assert image.dtype == dtype
        assert image.shape == (3, height, width)
        assert 0 <= image.min() <= 1
        assert 0 <= image.max() <= 1
        assert isinstance(objects, FrameLabels)
        assert objects.boxes.device == device
        assert objects.class_ids.device == device
        assert objects.boxes.dtype == dtype
        assert objects.class_ids.dtype == torch.int
        assert objects.boxes.shape == (1, 4)
        assert objects.class_ids.shape == (1,)
