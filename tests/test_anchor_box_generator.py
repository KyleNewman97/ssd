import torch

from torch import Tensor

from ssd.anchor_box_generator import AnchorBoxGenerator


class TestAnchorBoxGenerator:
    def test_init(self):
        """
        Test that we can initialise the AnchorBoxGenerator.
        """
        anchor_generator = AnchorBoxGenerator(torch.device("cpu"), torch.float32)
        assert isinstance(anchor_generator, AnchorBoxGenerator)

    def test_generate_wh_pairs(self):
        """
        Test that we can generate the width-height pairs for the anchor boxes.
        """
        anchor_generator = AnchorBoxGenerator(torch.device("cpu"), torch.float32)
        wh_pairs = anchor_generator._generate_wh_pairs()

        assert len(wh_pairs) == anchor_generator.num_feature_maps
        for k, _wh_pairs in enumerate(wh_pairs):
            num_scales = anchor_generator.aspect_ratios[k].shape[0]
            assert _wh_pairs.shape == (num_scales, 2)

    def _calculate_num_expected_anchor_boxes(
        self,
        feature_map_sizes: list[tuple[int, int]],
        all_aspect_ratios: list[Tensor],
    ) -> int:
        """
        Calculates the expected number of anchor boxes to be generated.
        """
        expected_boxes = 0
        for grid_size, aspect_ratios in zip(feature_map_sizes, all_aspect_ratios):
            expected_boxes += grid_size[0] * grid_size[1] * len(aspect_ratios)

        return expected_boxes

    def test_generate_anchor_boxes(self):
        """
        Test that we can create the default grid boxes.
        """
        feature_map_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        dtype = torch.float32

        anchor_generator = AnchorBoxGenerator(torch.device("cpu"), dtype)
        boxes = anchor_generator._generate_anchor_boxes(feature_map_sizes)

        assert isinstance(boxes, torch.Tensor)

        # Calculate the expected number of anchor boxes
        expected_boxes = self._calculate_num_expected_anchor_boxes(
            feature_map_sizes, anchor_generator.aspect_ratios
        )

        assert boxes.shape == (expected_boxes, 4)

        # Ensure the anchor box centroids are within the bounds of the image
        assert 0 < boxes[:, 0].min() and boxes[:, 0].max() < 1
        assert 0 < boxes[:, 1].min() and boxes[:, 1].max() < 1

    def test_forward(self):
        """
        Test that we can run forward inference on anchor box generation.
        """
        batch_size = 8
        feature_map_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        dtype = torch.float32

        anchor_generator = AnchorBoxGenerator(torch.device("cpu"), dtype)
        boxes = anchor_generator.forward(batch_size, feature_map_sizes)

        # Calculate the expected number of anchor boxes
        expected_boxes = self._calculate_num_expected_anchor_boxes(
            feature_map_sizes, anchor_generator.aspect_ratios
        )

        assert boxes.shape == (batch_size, expected_boxes, 4)

        # Ensure the anchor box centroids are within the bounds of the image
        image_boxes = boxes[0]
        assert 0 < image_boxes[:, 0].min() and image_boxes[:, 0].max() < 1
        assert 0 < image_boxes[:, 1].min() and image_boxes[:, 1].max() < 1
