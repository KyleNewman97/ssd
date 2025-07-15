import pytest
import torch

from ssd.structs import FrameDetections, FrameLabels
from ssd.utils import Metrics


class TestMetrics:
    def test_init(self):
        """
        Test that we can initialise a Metrics object.
        """
        metrics = Metrics()
        assert isinstance(metrics, Metrics)

    def _create_detections(
        self, boxes: list[list[float]], scores: list[float], class_ids: list[int]
    ) -> FrameDetections:
        """
        Creates detections with the specified inputs.
        """
        dtype = torch.float32
        device = torch.device("cpu")
        return FrameDetections(
            boxes=torch.tensor(boxes, dtype=dtype, device=device),
            scores=torch.tensor(scores, dtype=dtype, device=device),
            class_ids=torch.tensor(class_ids, dtype=torch.int, device=device),
        )

    def _create_labels(
        self, boxes: list[list[float]], class_ids: list[int]
    ) -> FrameLabels:
        """
        Creates labels with the specified inputs.
        """
        dtype = torch.float32
        device = torch.device("cpu")
        return FrameLabels(
            boxes=torch.tensor(boxes, dtype=dtype, device=device),
            class_ids=torch.tensor(class_ids, dtype=torch.int, device=device),
        )

    def test_frame_true_positives_with_tp(self):
        """
        Test that we can correctly identify when there is a true positive.
        """
        # Create dummy data
        num_classes = 20
        detections = self._create_detections([[0.1, 0.1, 0.1, 0.1]], [1], [0])
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0])

        tps = Metrics.frame_true_positives(detections, labels, num_classes, 0.5, 0.24)

        assert isinstance(tps, dict)
        assert list(tps.keys()) == [i for i in range(num_classes)]
        assert tps[0] == 1

    def test_frame_true_positives_insufficient_iou(self):
        """
        Test that we can identify when there are no true positives because insufficient
        overlap.
        """
        # Create dummy data
        num_classes = 20
        detections = self._create_detections([[0.1, 0.1, 0.1, 0.1]], [1], [0])
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0])

        tps = Metrics.frame_true_positives(detections, labels, num_classes, 0.5, 0.26)

        assert isinstance(tps, dict)
        assert list(tps.keys()) == [i for i in range(num_classes)]
        assert tps[0] == 0

    def test_frame_true_positives_insufficient_confidence(self):
        """
        Test that we can identify when there are no true positives when detections have
        insufficient confidence.
        """
        # Create dummy data
        num_classes = 20
        detections = self._create_detections([[0.1, 0.1, 0.1, 0.1]], [0.5], [0])
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0])

        tps = Metrics.frame_true_positives(detections, labels, num_classes, 0.6, 0.24)

        assert isinstance(tps, dict)
        assert list(tps.keys()) == [i for i in range(num_classes)]
        assert tps[0] == 0

    def test_frame_true_positives_different_classes(self):
        """
        Test that we get no true positives if a detection overlaps with a label of a
        different class.
        """
        # Create dummy data
        num_classes = 20
        detections = self._create_detections([[0.1, 0.1, 0.1, 0.1]], [1], [0])
        labels = self._create_labels([[0.1, 0.1, 0.1, 0.1]], [1])

        tps = Metrics.frame_true_positives(detections, labels, num_classes, 0.5, 0.5)

        assert isinstance(tps, dict)
        assert list(tps.keys()) == [i for i in range(num_classes)]
        assert tps[0] == 0
        assert tps[1] == 0

    def test_frame_true_positives_complex(self):
        """
        Test frame_true_positives with representative "complex" detections and labels.
        """
        # Create dummy data
        num_classes = 20
        detections = self._create_detections(
            [[0.1, 0.1, 0.05, 0.05], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.08, 0.08]],
            [1, 1, 1],
            [0, 1, 0],
        )
        labels = self._create_labels(
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.06, 0.06]], [0, 0]
        )

        tps = Metrics.frame_true_positives(detections, labels, num_classes, 0.5, 0.24)

        assert isinstance(tps, dict)
        assert list(tps.keys()) == [i for i in range(num_classes)]
        assert tps[0] == 2
        assert tps[1] == 0

    def test_frame_false_positives_with_tp(self):
        """
        Test we get no false positives when only true positives exists.
        """
        # Create dummy data
        num_classes = 20
        detections = self._create_detections([[0.1, 0.1, 0.1, 0.1]], [1], [0])
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0])

        fps = Metrics.frame_false_positives(detections, labels, num_classes, 0.5, 0.24)

        assert isinstance(fps, dict)
        assert list(fps.keys()) == [i for i in range(num_classes)]
        assert fps[0] == 0

    def test_frame_false_positives_complex(self):
        """
        Test we get the right number of false positives on a representative complex
        input.
        """
        # Create dummy data
        num_classes = 20
        detections = self._create_detections(
            [[0.1, 0.1, 0.05, 0.05], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.08, 0.08]],
            [1, 1, 1],
            [0, 1, 0],
        )
        labels = self._create_labels(
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.06, 0.06]], [0, 0]
        )

        fps = Metrics.frame_false_positives(detections, labels, num_classes, 0.5, 0.24)

        assert isinstance(fps, dict)
        assert list(fps.keys()) == [i for i in range(num_classes)]
        assert fps[0] == 0
        assert fps[1] == 1

    def test_frame_false_negatives_with_tp(self):
        """
        Test we get no false negatives when all labels are matched.
        """
        # Create dummy data
        num_classes = 20
        detections = self._create_detections([[0.1, 0.1, 0.1, 0.1]], [1], [0])
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0])

        fns = Metrics.frame_false_negatives(detections, labels, num_classes, 0.5, 0.24)

        assert isinstance(fns, dict)
        assert list(fns.keys()) == [i for i in range(num_classes)]
        assert fns[0] == 0

    def test_frame_false_negatives_complex(self):
        """
        Test we get the right number of false negatives on a representative complex
        input.
        """
        # Create dummy data
        num_classes = 20
        detections = self._create_detections(
            [[0.1, 0.1, 0.05, 0.05], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.08, 0.08]],
            [1, 1, 1],
            [0, 1, 0],
        )
        labels = self._create_labels(
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.06, 0.06], [0.1, 0.1, 0.1, 0.1]],
            [0, 0, 2],
        )

        fns = Metrics.frame_false_negatives(detections, labels, num_classes, 0.5, 0.24)

        assert isinstance(fns, dict)
        assert list(fns.keys()) == [i for i in range(num_classes)]
        assert fns[0] == 0
        assert fns[1] == 0
        assert fns[2] == 1
