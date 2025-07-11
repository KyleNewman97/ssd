import torch
from torch import Tensor


class TrainUtils:
    @staticmethod
    def batch_collate_func(
        batch: list[tuple[Tensor, Tensor]], device: torch.device
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Collates image and label pairs into a batch.

        Parameters
        ----------
        batch:
            All image and label pairs within the batch. This is structured as:
            `[(image, label), ...]`.

        device:
            The device the tensors should be on.

        Returns
        -------
        image_batch:
            A tensor of all the images in the batch. This will have dimensions:
            `(batch_size, num_channels, height, width)`.

        label_batch:
            A list of tensors. Each element corresponds to an image's ground truth
            objects. A list is used here to support variable number of objects per
            image. The shape of each tensor is `(num_objects, 5)`.
        """

        image_batch = torch.stack([el[0] for el in batch], dim=0).to(device=device)
        label_batch = [el[1] for el in batch]

        return image_batch, label_batch
