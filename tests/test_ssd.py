from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ssd import SSD
from ssd.anchor_box_generator import AnchorBoxGenerator


class TestSSD:
    def test_init(self):
        """
        Test the model can be initialised.
        """
        model = SSD(2)
        assert isinstance(model, SSD)

    def test_save_load(self):
        """
        Test that we can save and load the model in.
        """
        num_classes = 4
        device = torch.device("cpu")
        model = SSD(num_classes, device)

        with TemporaryDirectory() as temp_dir:
            file = Path(temp_dir) / "model.pt"

            # Try to save the model
            model.save(file)
            assert file.exists()

            # Try to load the model
            loaded_model = model.load(file)
            assert isinstance(loaded_model, SSD)
            assert loaded_model.num_classes == num_classes

    def test_post_process_detections_representative(self):
        """
        Test that the model can correctly post process detections on a random
        representative input tensor.
        """

        # Initialise the model
        num_classes = 2
        model = SSD(num_classes)

        # Initialise dummy inputs
        batch_size = 2
        feature_map_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        head_outputs = torch.rand(
            (batch_size, 8732, 4 + 1 + num_classes), device=model.device
        )
        anchors = AnchorBoxGenerator(torch.device("cpu")).forward(
            batch_size, feature_map_sizes
        )

        # Try to post-process the detections
        confidence_threshold = 0.5
        num_top_k = 4
        nms_iou_threshold = 0.3
        all_frame_detections = model._post_process_detections(
            head_outputs, anchors, confidence_threshold, num_top_k, nms_iou_threshold
        )

        assert isinstance(all_frame_detections, list)
        assert len(all_frame_detections) == batch_size

        for frame_detections in all_frame_detections:
            assert frame_detections.boxes.shape == (num_top_k, 4)
            assert frame_detections.scores.shape == (num_top_k,)
            assert frame_detections.labels.shape == (num_top_k,)

            assert frame_detections.scores.min() > confidence_threshold
