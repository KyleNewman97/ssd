import pytest
import torch
from torch import Tensor

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
            class_ids_with_background=torch.tensor(
                class_ids, dtype=torch.int, device=device
            ),
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

        assert isinstance(tps, Tensor)
        assert tps.shape == (num_classes,)
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

        assert isinstance(tps, Tensor)
        assert tps.shape == (num_classes,)
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

        assert isinstance(tps, Tensor)
        assert tps.shape == (num_classes,)
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

        assert isinstance(tps, Tensor)
        assert tps.shape == (num_classes,)
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

        assert isinstance(tps, Tensor)
        assert tps.shape == (num_classes,)
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

        assert isinstance(fps, Tensor)
        assert fps.shape == (num_classes,)
        assert fps[0] == 0

    def test_frame_false_positives_below_confidence(self):
        """
        Test that detections below the confidence threshold do not get counted as false
        positives.
        """
        # Create dummy data
        num_classes = 20
        detections = self._create_detections([[0.1, 0.1, 0.1, 0.1]], [0.5], [0])
        labels = self._create_labels([[0.1, 0.1, 0.1, 0.1]], [0])

        fps = Metrics.frame_false_positives(detections, labels, num_classes, 0.6, 0.24)

        assert isinstance(fps, Tensor)
        assert fps.shape == (num_classes,)
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

        assert isinstance(fps, Tensor)
        assert fps.shape == (num_classes,)
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

        assert isinstance(fns, Tensor)
        assert fns.shape == (num_classes,)
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

        assert isinstance(fns, Tensor)
        assert fns.shape == (num_classes,)
        assert fns[0] == 0
        assert fns[1] == 0
        assert fns[2] == 1

    def test_precision(self):
        """
        Test we can calculate the precision correctly.
        """
        device = torch.device("cpu")
        tps = torch.tensor([0, 1, 2, 3], dtype=torch.int, device=device)
        fps = torch.tensor([0, 0, 1, 4], dtype=torch.int, device=device)

        precisions = Metrics.precision(tps, fps)

        assert isinstance(precisions, Tensor)
        assert precisions.shape == tps.shape
        assert precisions.dtype == torch.float32
        expected = torch.tensor(
            [1, 1, 2 / 3, 3 / 7], dtype=torch.float32, device=device
        )
        assert precisions.allclose(expected)

    def test_precision_mismatching_shapes(self):
        """
        Test we get an error if the TPs and FPs have different shapes.
        """
        device = torch.device("cpu")
        tps = torch.randint(0, 100, (20,), device=device)
        fps = torch.randint(0, 100, (18,), device=device)

        with pytest.raises(ValueError):
            Metrics.precision(tps, fps)

    def test_recall(self):
        """
        Test we can calculate the recall correctly.
        """
        device = torch.device("cpu")
        tps = torch.tensor([0, 1, 2, 3], dtype=torch.int, device=device)
        fns = torch.tensor([0, 0, 1, 4], dtype=torch.int, device=device)

        recalls = Metrics.recall(tps, fns)

        assert isinstance(recalls, Tensor)
        assert recalls.shape == tps.shape
        assert recalls.dtype == torch.float32
        expected = torch.tensor(
            [1, 1, 2 / 3, 3 / 7], dtype=torch.float32, device=device
        )
        assert recalls.allclose(expected)

    def test_recall_mismatching_shapes(self):
        """
        Test we get an error if the TPs and FNs have different shapes.
        """
        device = torch.device("cpu")
        tps = torch.randint(0, 100, (20,), device=device)
        fns = torch.randint(0, 100, (18,), device=device)

        with pytest.raises(ValueError):
            Metrics.recall(tps, fns)
