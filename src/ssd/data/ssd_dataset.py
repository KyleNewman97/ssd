from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from ssd.data.letterbox_transform import LetterboxTransform
from ssd.utils import MetaLogger


class SSDDataset(Dataset, MetaLogger):
    def __init__(
        self,
        images_path: Path,
        labels_path: Path,
        num_classes: int,
        transform: LetterboxTransform | None,
    ):
        """
        Parameters
        ----------
        images_path:
            Path to a subset (train, valid, test) of the image data.

        labels_path:
            Path to a subset (train, valid, test) of the label data.

        num_classes:
            The number of classes classification will be run on.

        transform:
            The transform to apply to the images after they are loaded in.
        """
        Dataset.__init__(self)
        MetaLogger.__init__(self)

        self.images_path = images_path
        self.labels_path = labels_path
        self.num_classes = num_classes
        self.transform = transform

        self.logger.info(f"Images path: {images_path}")
        self.logger.info(f"Labels path: {labels_path}")

        images = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        labels = list(labels_path.glob("*.txt"))

        self.logger.info(f"Found {len(images)} images and {len(labels)} labels.")

        self.samples = self._get_file_pairs(images, labels)
        self.samples = self._filter_for_valid_labels(self.samples)

        self.logger.info(f"{len(self.samples)} image and label pairs exist.")

    def _get_file_pairs(
        self, images: list[Path], labels: list[Path]
    ) -> list[tuple[Path, Path]]:
        """
        Keeps only image and label files that form a pair. A pair requires the image
        and label file to have the same stem.

        Output
        ------
        samples:
            A list of samples. Each sample is structured as `(image_file, label_file)`.
        """
        stem_to_image_file = {i.stem: i for i in images}
        stem_to_label_file = {l.stem: l for l in labels}

        samples: list[tuple[Path, Path]] = []

        for stem, image_file in stem_to_image_file.items():
            if stem not in stem_to_label_file:
                self.logger.warning(f"Missing label file for {image_file}.")
                continue
            samples.append((image_file, stem_to_label_file[stem]))

        return samples

    def _filter_for_valid_labels(
        self, samples: list[tuple[Path, Path]]
    ) -> list[tuple[Path, Path]]:
        """
        Filter out samples with invalid labels.
        """
        expected_classes = set([i for i in range(self.num_classes)])

        filtered_samples: list[tuple[Path, Path]] = []
        for sample in samples:
            with open(sample[1], "r") as fp:
                # Read the label file and ensure everything is correct
                is_valid = True
                for line in fp.readlines():
                    elements = line.strip().split(" ")

                    # Check that there are five elements in the label
                    if len(elements) != 5:
                        msg = f"[{sample[1]}] {len(elements)} observed but require 5."
                        self.logger.warning(msg)
                        is_valid = False
                        continue

                    # Check the class is correct
                    if int(elements[0]) not in expected_classes:
                        msg = f"[{sample[1]}] invalid class {int(elements[0])}."
                        self.logger.warning(msg)
                        is_valid = False
                        continue

                    # Check bounding box is valid
                    for element in elements[1:]:
                        value = float(element)
                        if value < 0 or 1 < value:
                            msg = f"[{sample[1]}] invalid box coord {value}."
                            self.logger.warning(msg)
                            is_valid = False
                            continue

                if is_valid:
                    filtered_samples.append(sample)

        return filtered_samples

    @staticmethod
    def read_label_file(file: Path) -> Tensor:
        """
        Reads in a label file and returns a tensor of the objects contained in it.

        Parameters
        ----------
        file:
            Path to the label file. The contents of this file should be structured as:

            ```
            class_id cx cy w h
            class_id cx cy w h
            ...
            ```

            Where the box coords should be defined in the normalised image space
            (between 0 and 1).

        Returns
        -------
        objects:
            A tensor of the objects defined in the label file. This will have dimensions
            of `(num_objects, 5)` and will have the elements ordered as:

            `(class_id, cx, cy, w, h)`
        """
        # Load in the labels
        with open(file, "r") as fp:
            lines = fp.read().split("\n")
            labels: list[tuple[int, float, float, float, float]] = []
            for line in lines:
                elements = line.strip().split(" ")

                # Only allow valid label rows
                if len(elements) != 5:
                    continue

                class_id = int(elements[0])
                center_x = float(elements[1])
                center_y = float(elements[2])
                width = float(elements[3])
                height = float(elements[4])
                labels.append((class_id, center_x, center_y, width, height))

        # Put the labels into a tensor
        label_tensor = torch.zeros((len(labels), 5), dtype=torch.float32)
        for idx, label in enumerate(labels):
            label_tensor[idx, :] = torch.tensor(label)

        return label_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image_file, label_file = self.samples[idx]

        # Load in the image and pre-process it
        pil_image = Image.open(image_file)
        image = pil_to_tensor(pil_image)
        objects = SSDDataset.read_label_file(label_file)

        if self.transform is None:
            return image, objects
        else:
            return self.transform(image, objects)
