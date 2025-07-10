from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
import numpy as np
from PIL import Image
from torch import Tensor

from ssd.data import LetterboxTransform, SSDDataset


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

    def test_init_valid_dataset(self):
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
        dataset = SSDDataset(self.images_path, self.labels_path, 2, transform)
        assert isinstance(dataset, SSDDataset)
        assert len(dataset) == 1

    def test_init_missing_label(self):
        """
        Test that we remove samples when we are missing the label.
        """

        # Make a single image file
        uuid = f"{uuid4()}"

        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")

        # Try to initialise the dataset
        transform = LetterboxTransform()
        dataset = SSDDataset(self.images_path, self.labels_path, 2, transform)
        assert isinstance(dataset, SSDDataset)
        assert len(dataset) == 0

    def test_init_invalid_label_num_elements(self):
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
        dataset = SSDDataset(self.images_path, self.labels_path, 2, transform)
        assert isinstance(dataset, SSDDataset)
        assert len(dataset) == 0

    def test_init_invalid_label_class(self):
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
        dataset = SSDDataset(self.images_path, self.labels_path, 2, transform)
        assert isinstance(dataset, SSDDataset)
        assert len(dataset) == 0

    def test_init_invalid_label_box(self):
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
        dataset = SSDDataset(self.images_path, self.labels_path, 2, transform)
        assert isinstance(dataset, SSDDataset)
        assert len(dataset) == 0

    def test_read_label_file(self):
        """
        Test that we can read in a valid label file.
        """
        with TemporaryDirectory() as temp_dir:
            # Create a dummy label file
            file = Path(temp_dir) / "label.txt"
            with open(file, "w") as fp:
                fp.write("0 0.5358 0.4942 0.3483 0.7085\n1 0.2 0.4 0.02 0.08\n")

            # Try to read in the labels
            labels = SSDDataset.read_label_file(file)

        # Check the contents is correct
        assert labels.shape == (2, 5)

    def test_test_get_item(self):
        """
        Test that we can get an item from the dataset.
        """

        # Make a single image file and label file that are valid
        uuid = f"{uuid4()}"

        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")

        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("0 0.1 0.1 0.1 0.1")

        # Try to get a sample from the dataset
        num_classes = 2
        height, width = 256, 512
        transform = LetterboxTransform(height=height, width=width)
        dataset = SSDDataset(self.images_path, self.labels_path, num_classes, transform)
        image, label = dataset[0]

        # Check the output shapes are correct
        assert isinstance(image, Tensor)
        assert image.shape == (3, height, width)
        assert isinstance(label, Tensor)
        assert label.shape == (1, 5)
