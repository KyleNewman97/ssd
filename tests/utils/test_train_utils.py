import torch
from torch import Tensor

from ssd.utils import TrainUtils


class TestTrainUtils:
    def test_batch_collate_func(self):
        """
        Test that the collate function works correctly.
        """
        device = torch.device("cpu")

        # Construct a dummy batch
        batch: list[tuple[Tensor, Tensor]] = []
        batch_size = 16
        image_shape = (3, 300, 300)
        for _ in range(batch_size):
            batch.append(
                (
                    torch.rand(image_shape, device=device),
                    torch.rand((2, 5), device=device),
                )
            )

        image_batch, label_batch = TrainUtils.batch_collate_func(batch, device)

        assert isinstance(image_batch, Tensor)
        assert image_batch.device == device
        assert image_batch.shape == (
            batch_size,
            image_shape[0],
            image_shape[1],
            image_shape[2],
        )
        assert isinstance(label_batch, list)
        assert len(label_batch) == batch_size
        assert label_batch[0].device == device
