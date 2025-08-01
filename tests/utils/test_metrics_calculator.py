import torch
from pytest_mock import MockerFixture
from torch import Tensor

from ssd.structs import FrameDetections, FrameLabels
from ssd.utils import MetricsCalculator


class TestMetricsCalculator:
    def test_init(self):
        """
        Test that we can initialise a MetricsCalculator object.
        """
        num_classes = 20
        metrics = MetricsCalculator(num_classes)
        assert isinstance(metrics, MetricsCalculator)

    def _create_detections(
        self,
        boxes: list[list[float]],
        scores: list[float],
        class_ids: list[int],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> FrameDetections:
        """
        Creates detections with the specified inputs.
        """
        return FrameDetections(
            boxes=torch.tensor(boxes, dtype=dtype, device=device),
            scores=torch.tensor(scores, dtype=dtype, device=device),
            class_ids=torch.tensor(class_ids, dtype=torch.int, device=device),
        )

    def _create_labels(
        self,
        boxes: list[list[float]],
        class_ids: list[int],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> FrameLabels:
        """
        Creates labels with the specified inputs.
        """

        # Convert class IDs to include a background class
        class_id_with_background = [c + 1 for c in class_ids]

        return FrameLabels(
            boxes=torch.tensor(boxes, dtype=dtype, device=device),
            class_ids_with_background=torch.tensor(
                class_id_with_background, dtype=torch.int, device=device
            ),
        )

    def test_update(self):
        """
        Test that the update method works as expected with simple inputs.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        num_classes = 20
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1]], [0.5], [0], dtype, device
        )
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0], dtype, device)

        # Try to update the state
        metrics = MetricsCalculator(num_classes)
        metrics.update([detections], [labels])

        # Check IoU state was updated correctly
        assert isinstance(metrics._frame_per_class_ious, list)
        assert len(metrics._frame_per_class_ious) == 1

        per_class_ious = metrics._frame_per_class_ious[0]
        assert isinstance(per_class_ious, dict)
        assert list(per_class_ious.keys()) == [0]
        assert per_class_ious[0].shape == (1, 1)

        expected_ious = torch.tensor([0.25], dtype=dtype, device=device)
        assert per_class_ious[0].allclose(expected_ious)

        # Check the scores state was updated correctly
        assert isinstance(metrics._frame_per_class_scores, list)
        assert len(metrics._frame_per_class_scores) == 1

        per_class_scores = metrics._frame_per_class_scores[0]
        assert isinstance(per_class_scores, dict)
        assert list(per_class_scores.keys()) == [0]

        expected_scores = torch.tensor([0.5], dtype=dtype, device=device)
        assert per_class_scores[0].allclose(expected_scores)

        # Check the label count state was updated correctly
        assert isinstance(metrics._frame_per_class_label_count, list)
        assert len(metrics._frame_per_class_label_count) == 1

        per_class_label_count = metrics._frame_per_class_label_count[0]
        assert isinstance(per_class_label_count, dict)
        assert list(per_class_label_count.keys()) == [0]
        assert per_class_label_count[0] == 1

    def test_update_mismatching_classes(self):
        """
        Test that the update method works with mismatching classes.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        num_classes = 20
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1]], [0.5], [0], dtype, device
        )
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [1], dtype, device)

        # Try to update the state
        metrics = MetricsCalculator(num_classes)
        metrics.update([detections], [labels])

        # Check we get zero IoU for different classes
        observed_ious = metrics._frame_per_class_ious[0][0]
        assert observed_ious.shape == (1, 0)

        # Check we get the right score saved
        observed_scores = metrics._frame_per_class_scores[0][0]
        expected_scores = torch.tensor([0.5], dtype=dtype, device=device)
        assert observed_scores.allclose(expected_scores)

        # Check we get the right label count saved
        observed_count = metrics._frame_per_class_label_count[0][1]
        assert observed_count == 1

    def test_tps_single_det_and_lab(self):
        """
        Test we can calculate the TPs correctly when we have a single detection and
        label.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1]], [0.5], [0], dtype, device
        )
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0], dtype, device)

        # Calculate the number of true positives
        num_classes = 1
        iou_thresholds = [0.24]
        conf_thresholds = [0.4]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        tps = metrics.tps()

        # Check the output is correct
        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert tps.shape == expected_shape
        expected_tps = torch.ones(expected_shape, dtype=torch.int, device=device)
        assert tps.equal(expected_tps)

    def test_tps_multiple_dets_and_labs(self):
        """
        Test we can calculate the TPs correctly when we have multiple detections and
        labels.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1]],
            [0.5, 0.5, 0.5],
            [0, 0, 0],
            dtype,
            device,
        )
        labels = self._create_labels(
            [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.05, 0.05], [0.3, 0.3, 0.08, 0.08]],
            [0, 0, 0],
            dtype,
            device,
        )

        # Calculate the number of true positives
        num_classes = 1
        iou_thresholds = [0.24]
        conf_thresholds = [0.4]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        tps = metrics.tps()

        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert tps.shape == expected_shape
        expected_tps = torch.ones(expected_shape, dtype=torch.int, device=device) * 3
        assert tps.equal(expected_tps)

    def test_tps_insufficient_iou(self):
        """
        Test we get no TPs when the overlap is insufficient.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1]], [0.5], [0], dtype, device
        )
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0], dtype, device)

        # Calculate the number of true positives
        num_classes = 1
        iou_thresholds = [0.26]
        conf_thresholds = [0.4]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        tps = metrics.tps()

        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert tps.shape == expected_shape
        expected_tps = torch.zeros(expected_shape, dtype=torch.int, device=device)
        assert tps.equal(expected_tps)

    def test_tps_insufficient_confidence(self):
        """
        Test we get no TPs when the confidence is insufficient.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1]], [0.5], [0], dtype, device
        )
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0], dtype, device)

        # Calculate the number of true positives
        num_classes = 1
        iou_thresholds = [0.24]
        conf_thresholds = [0.6]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        tps = metrics.tps()

        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert tps.shape == expected_shape
        expected_tps = torch.zeros(expected_shape, dtype=torch.int, device=device)
        assert tps.equal(expected_tps)

    def test_tps_different_classes(self):
        """
        Test we get no TPs when the objects are of different classes.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1]], [0.5], [0], dtype, device
        )
        labels = self._create_labels([[0.1, 0.1, 0.1, 0.1]], [1], dtype, device)

        # Calculate the number of true positives
        num_classes = 2
        iou_thresholds = [0.24]
        conf_thresholds = [0.4]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        tps = metrics.tps()

        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert tps.shape == expected_shape
        expected_tps = torch.zeros(expected_shape, dtype=torch.int, device=device)
        assert tps.equal(expected_tps)

    def test_tps_multi_detection_association(self):
        """
        Test that a detection is still counted as a TP if the GT box it has the highest
        IoU with is already used and there is another GT box that it overlaps with (with
        the overlap being over the IoU threshold).
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.08, 0.08]],
            [0.5, 0.5],
            [0, 0],
            dtype,
            device,
        )
        labels = self._create_labels(
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.05, 0.05]], [0, 0], dtype, device
        )

        # Calculate the number of true positives
        num_classes = 1
        iou_thresholds = [0.24]
        conf_thresholds = [0.4]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        tps = metrics.tps()

        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert tps.shape == expected_shape
        expected_tps = torch.ones(expected_shape, dtype=torch.int, device=device) * 2
        assert tps.equal(expected_tps)

    def test_tps_multi_label_association(self):
        """
        Test that we still count a TP if a label's highest IoU detection is already
        matched but there is another detection that overlaps it above the IoU threshold.
        """

        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.05, 0.05]],
            [0.5, 0.5],
            [0, 0],
            dtype,
            device,
        )
        labels = self._create_labels(
            [[0.1, 0.1, 0.08, 0.08], [0.1, 0.1, 0.1, 0.1]], [0, 0], dtype, device
        )

        # Calculate the number of true positives
        num_classes = 1
        iou_thresholds = [0.24]
        conf_thresholds = [0.4]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        tps = metrics.tps()

        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert tps.shape == expected_shape
        expected_tps = torch.ones(expected_shape, dtype=torch.int, device=device) * 2
        assert tps.equal(expected_tps)

    def test_tps_multi_conf_iou_and_classes(self):
        """
        Test we can calculate the TPs correctly with multiple confidence thresholds, IoU
        thresholds and classes.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1]], [0.5], [0], dtype, device
        )
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0], dtype, device)

        # Calculate the number of true positives
        num_classes = 2
        iou_thresholds = [0.24, 0.26]
        conf_thresholds = [0.4, 0.6]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        tps = metrics.tps()

        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert tps.shape == expected_shape
        expected_tps = torch.tensor(
            [[[1, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=torch.int, device=device
        )
        assert tps.equal(expected_tps)

    def test_fps_with_tp(self):
        """
        Test we get no false positives when only true positives exists.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1]], [0.5], [0], dtype, device
        )
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0], dtype, device)

        # Calculate the number of true positives
        num_classes = 1
        iou_thresholds = [0.24]
        conf_thresholds = [0.4]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        fps = metrics.fps()

        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert fps.shape == expected_shape
        expected_fps = torch.zeros(expected_shape, dtype=torch.int, device=device)
        assert fps.equal(expected_fps)

    def test_fps_below_confidence(self):
        """
        Test that detections below the confidence threshold do not get counted as false
        positives.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1]], [0.5], [0], dtype, device
        )
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0], dtype, device)

        # Calculate the number of true positives
        num_classes = 1
        iou_thresholds = [0.24]
        conf_thresholds = [0.6]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        fps = metrics.fps()

        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert fps.shape == expected_shape
        expected_fps = torch.zeros(expected_shape, dtype=torch.int, device=device)
        assert fps.equal(expected_fps)

    def test_fps_complex(self):
        """
        Test FP calculation with a complex representative input.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.08, 0.08]],
            [0.5, 0.6],
            [0, 0],
            dtype,
            device,
        )
        labels = self._create_labels(
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.04, 0.04]], [0, 0], dtype, device
        )

        # Calculate the number of true positives
        num_classes = 1
        iou_thresholds = [0.24, 0.26]
        conf_thresholds = [0.4, 0.55]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        fps = metrics.fps()

        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert fps.shape == expected_shape
        expected_fps = torch.tensor(
            [[[0], [1]], [[0], [0]]], dtype=torch.int, device=device
        )
        assert fps.equal(expected_fps)

    def test_fns_with_tp(self):
        """
        Test we get no false negatives when all labels are matched.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1]], [0.5], [0], dtype, device
        )
        labels = self._create_labels([[0.1, 0.1, 0.05, 0.05]], [0], dtype, device)

        # Calculate the number of true positives
        num_classes = 1
        iou_thresholds = [0.24]
        conf_thresholds = [0.4]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        fns = metrics.fns()

        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert fns.shape == expected_shape
        expected_fns = torch.zeros(expected_shape, dtype=torch.int, device=device)
        assert fns.equal(expected_fns)

    def test_fns_complex(self):
        """
        Test we get the right number of false negatives on a representative complex
        input.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        detections = self._create_detections(
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.08, 0.08]],
            [0.5, 0.6],
            [0, 0],
            dtype,
            device,
        )
        labels = self._create_labels(
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.04, 0.04]], [0, 0], dtype, device
        )

        # Calculate the number of true positives
        num_classes = 1
        iou_thresholds = [0.24, 0.26]
        conf_thresholds = [0.4, 0.55]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics.update([detections], [labels])
        fns = metrics.fns()

        expected_shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        assert fns.shape == expected_shape
        expected_fns = torch.tensor(
            [[[0], [1]], [[1], [1]]], dtype=torch.int, device=device
        )
        assert fns.equal(expected_fns)

    def test_precisions(self):
        """
        Test we can calculate the precisions correctly.
        """
        device = torch.device("cpu")

        num_classes = 2
        iou_thresholds = [0.24, 0.26]
        conf_thresholds = [0.4, 0.55]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics._device = device

        # Create dummy data
        shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        tps = torch.randint(0, 100, shape, dtype=torch.int, device=device)
        fps = torch.randint(0, 100, shape, dtype=torch.int, device=device)
        metrics._tps = tps
        metrics._fps = fps

        # Calculate the precisions
        precisions = metrics.precisions()

        assert isinstance(precisions, Tensor)
        assert precisions.shape == shape
        assert precisions.dtype == torch.float32

    def test_precisions_no_tp_or_fp(self):
        """
        Test that we get a 1 when `tp = 0` and `fp = 0`.
        """
        device = torch.device("cpu")

        num_classes = 1
        iou_thresholds = [0.24]
        conf_thresholds = [0.4]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics._device = device

        # Create dummy data
        tps = torch.tensor([[[0]]], dtype=torch.int, device=device)
        fps = torch.tensor([[[0]]], dtype=torch.int, device=device)
        metrics._tps = tps
        metrics._fps = fps

        # Calculate the precisions
        precisions = metrics.precisions()

        expected = torch.tensor([[[1]]], dtype=torch.float32, device=device)
        assert precisions.equal(expected)

    def test_recalls(self):
        """
        Test we can calculate the recalls correctly.
        """
        device = torch.device("cpu")

        num_classes = 2
        iou_thresholds = [0.24, 0.26]
        conf_thresholds = [0.4, 0.55]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics._device = device

        # Create dummy data
        shape = (len(conf_thresholds), len(iou_thresholds), num_classes)
        tps = torch.randint(0, 100, shape, dtype=torch.int, device=device)
        fns = torch.randint(0, 100, shape, dtype=torch.int, device=device)
        metrics._tps = tps
        metrics._fns = fns

        # Calculate the recalls
        recalls = metrics.recalls()

        assert isinstance(recalls, Tensor)
        assert recalls.shape == shape
        assert recalls.dtype == torch.float32

    def test_recalls_no_tp_or_fn(self):
        """
        Test that we get a 1 when `tp = 0` and `fn = 0`.
        """
        device = torch.device("cpu")

        num_classes = 1
        iou_thresholds = [0.24]
        conf_thresholds = [0.4]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        metrics._device = device

        # Create dummy data
        tps = torch.tensor([[[0]]], dtype=torch.int, device=device)
        fns = torch.tensor([[[0]]], dtype=torch.int, device=device)
        metrics._tps = tps
        metrics._fns = fns

        # Calculate the precisions
        recalls = metrics.recalls()

        expected = torch.tensor([[[1]]], dtype=torch.float32, device=device)
        assert recalls.equal(expected)

    def test_APs(self, mocker: MockerFixture):
        """
        Test that we can calculate the average precision correctly.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy precision and recalls
        precisions = torch.tensor(
            [[[0.0]], [[0.2]], [[0.4]], [[0.6]], [[0.8]], [[1.0]]],
            dtype=dtype,
            device=device,
        )
        recalls = torch.tensor(
            [[[1.0]], [[0.8]], [[0.6]], [[0.4]], [[0.2]], [[0.0]]],
            dtype=dtype,
            device=device,
        )
        mocker.patch.object(MetricsCalculator, "precisions", return_value=precisions)
        mocker.patch.object(MetricsCalculator, "recalls", return_value=recalls)

        # Calculate average precisions
        num_classes = 1
        iou_thresholds = [0.24]
        conf_thresholds = [0.1 * (i + 1) for i in range(6)]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        APs = metrics.APs()

        # Check the output is correct
        assert isinstance(APs, Tensor)
        expected_shape = (len(iou_thresholds), num_classes)
        assert APs.shape == expected_shape
        expected_APs = torch.tensor([[0.5]], dtype=dtype, device=device)
        assert APs.allclose(expected_APs)

    def test_mAPs(self, mocker: MockerFixture):
        """
        Test that we can calculate the mAP correctly.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy precision and recalls
        precisions = torch.tensor(
            [[[0.0]], [[0.2]], [[0.4]], [[0.6]], [[0.8]], [[1.0]]],
            dtype=dtype,
            device=device,
        )
        recalls = torch.tensor(
            [[[1.0]], [[0.8]], [[0.6]], [[0.4]], [[0.2]], [[0.0]]],
            dtype=dtype,
            device=device,
        )
        mocker.patch.object(MetricsCalculator, "precisions", return_value=precisions)
        mocker.patch.object(MetricsCalculator, "recalls", return_value=recalls)

        # Calculate mAP
        num_classes = 1
        iou_thresholds = [0.24]
        conf_thresholds = [0.1 * (i + 1) for i in range(6)]
        metrics = MetricsCalculator(num_classes, iou_thresholds, conf_thresholds)
        APs = metrics.APs()

        # Check the output is correct
        assert isinstance(APs, Tensor)
        expected_shape = (len(iou_thresholds), num_classes)
        assert APs.shape == expected_shape
        expected_APs = torch.tensor([0.5], dtype=dtype, device=device)
        assert APs.allclose(expected_APs)
