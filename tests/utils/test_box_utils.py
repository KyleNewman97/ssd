import torch
from torch import Tensor

from ssd.utils import BoxUtils


class TestBoxUtils:
    def test_regression_domain_to_image_domain(self):
        """
        Test that we can convert from regression domain to image domain.
        """

        # Create dummy data
        box_regressions = torch.tensor([[[1, 0, 0, -1]]])
        anchors = torch.tensor([[[10, 0, 20, 40]]])

        boxes = BoxUtils.regression_domain_to_image_domain(box_regressions, anchors)

        # Check the output is correct - the values are manually calculated using the
        # functions defined on page 5 of the SSD paper
        assert isinstance(boxes, torch.Tensor)
        assert boxes.shape == anchors.shape
        assert boxes.isclose(torch.tensor([[[30, 0, 20, 14.7152]]])).all()

    def test_regression_domain_to_image_domain_representative(self):
        """
        Test that we get no errors when working on representative input tensors.
        """

        # Create dummy data
        box_regressions = torch.rand((8, 8732, 4), dtype=torch.float32)
        anchors = torch.rand((8, 8732, 4), dtype=torch.float32)

        boxes = BoxUtils.regression_domain_to_image_domain(box_regressions, anchors)

        assert isinstance(boxes, torch.Tensor)
        assert boxes.shape == anchors.shape

    def test_image_domain_to_regression_domain(self):
        """
        Test that we can convert from image domain to regression domain.
        """

        # Create dummy data
        boxes = torch.tensor([[[30, 0, 20, 14.7152]]], dtype=torch.float32)
        anchors = torch.tensor([[[10, 0, 20, 40]]], dtype=torch.float32)

        regression_boxes = BoxUtils.image_domain_to_regression_domain(boxes, anchors)

        # Check the output is correct - the values are manually calculated using the
        # functions defined on page 5 of the SSD paper
        assert isinstance(regression_boxes, torch.Tensor)
        assert regression_boxes.shape == boxes.shape
        expected_result = torch.tensor([[[1, 0, 0, -1]]], dtype=torch.float32)
        assert regression_boxes.isclose(expected_result).all()

    def test_image_domain_to_regression_domain_representative(self):
        """
        Test that we get no errors when working on representative input tensors.
        """

        # Create dummy data
        boxes = torch.rand((8, 20, 4), dtype=torch.float32)
        anchors = torch.rand((8, 20, 4), dtype=torch.float32)

        regression_boxes = BoxUtils.image_domain_to_regression_domain(boxes, anchors)

        assert isinstance(regression_boxes, torch.Tensor)
        assert regression_boxes.shape == boxes.shape

    def test_find_indices_of_best_anchor_boxes(self):
        """
        Test that we can find the anchor boxes that are closest to the ground truth
        boxes.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Construct dummy inputs
        anchors = torch.tensor(
            [[[10, 10, 6, 6], [100, 100, 60, 60]]], dtype=dtype, device=device
        )
        labels = [torch.tensor([[88, 120, 40, 40]], dtype=dtype, device=device)]

        # Try to find the best anchor box indices
        best_anchor_indices, gt_indices = BoxUtils.find_indices_of_best_anchor_boxes(
            anchors, labels
        )

        # Check the output is correct
        assert isinstance(best_anchor_indices, list)
        assert len(best_anchor_indices) == 1
        assert best_anchor_indices[0] == torch.tensor([1])
        assert isinstance(gt_indices, list)
        assert len(gt_indices) == 1
        assert gt_indices[0].equal(torch.tensor([0], dtype=torch.int, device=device))

    def test_find_indices_of_best_anchor_boxes_representative(self):
        """
        Test that we can run the function with representative inputs.
        """

        # Construct dummy inputs
        batch_size = 8
        num_objects = 20
        anchors = torch.rand((batch_size, 8732, 4))
        ground_truth = [torch.rand((num_objects, 4))] * batch_size

        # Try to find the best anchor box indices
        best_anchor_indices, gt_indices = BoxUtils.find_indices_of_best_anchor_boxes(
            anchors, ground_truth
        )

        # Check the output is correct
        assert isinstance(best_anchor_indices, list)
        assert len(best_anchor_indices) == batch_size
        assert best_anchor_indices[0].shape == (num_objects,)
        assert isinstance(gt_indices, list)
        assert len(gt_indices) == batch_size
        assert gt_indices[0].shape == (num_objects,)

    def test_find_indicies_of_high_out_anchors(self):
        """
        Test that we can find the indicies of anchors (and the corresponding ground
        truth boxes) that have an IoU above the specified threshold.
        """
        # Construct dummy inputs
        anchors = torch.tensor(
            [
                [
                    [0.05, 0.05, 0.1, 0.1],
                    [0.06, 0.06, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1],
                    [2, 2, 0.1, 0.1],
                ]
            ]
        )
        labels = [
            torch.tensor(
                [[0.05, 0.05, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [1, 1, 0.1, 0.1]]
            )
        ]

        anchor_idxs, gt_idxs = BoxUtils.find_indices_of_high_iou_anchors(
            anchors, labels, 0.1
        )

        assert isinstance(anchor_idxs, list)
        assert len(anchor_idxs) == anchors.shape[0]
        assert anchor_idxs[0].equal(torch.tensor([0, 1, 2]))

        assert isinstance(gt_idxs, list)
        assert len(gt_idxs) == anchors.shape[0]
        assert gt_idxs[0].equal(torch.tensor([0, 0, 1]))

    def test_nms_different_classes(self):
        """
        Test that NMS does not filter out boxes of different classes.
        """
        # Create dummy data
        device = torch.device("cpu")
        dtype = torch.float32
        in_boxes = torch.tensor(
            ([[0.05, 0.05, 0.1, 0.1], [0.05, 0.05, 0.1, 0.1]]),
            dtype=dtype,
            device=device,
        )
        in_scores = torch.tensor([1, 1], dtype=dtype, device=device)
        in_labels = torch.tensor([0, 1], dtype=torch.int, device=device)

        boxes, scores, labels = BoxUtils.nms(in_boxes, in_scores, in_labels, 0.1)

        assert isinstance(boxes, Tensor)
        assert boxes.shape == in_boxes.shape
        assert boxes.equal(in_boxes)

        assert isinstance(scores, Tensor)
        assert scores.shape == in_scores.shape
        assert scores.equal(in_scores)

        assert isinstance(labels, Tensor)
        assert labels.shape == in_labels.shape
        assert labels.equal(in_labels)

    def test_nms_same_class(self):
        """
        Test that NMS filters boxes of the same class (above the IoU threshold).
        """
        # Create dummy data
        device = torch.device("cpu")
        dtype = torch.float32
        in_boxes = torch.tensor(
            ([[0.05, 0.05, 0.1, 0.1], [0.05, 0.05, 0.05, 0.05]]),
            dtype=dtype,
            device=device,
        )
        in_scores = torch.tensor([1, 1], dtype=dtype, device=device)
        in_labels = torch.tensor([0, 0], dtype=torch.int, device=device)

        boxes, scores, labels = BoxUtils.nms(in_boxes, in_scores, in_labels, 0.24)

        assert isinstance(boxes, Tensor)
        expected_boxes = in_boxes[:1, :]
        assert boxes.shape == expected_boxes.shape
        assert boxes.equal(expected_boxes)

        assert isinstance(scores, Tensor)
        expected_scores = in_scores[:1]
        assert scores.shape == expected_scores.shape
        assert scores.equal(expected_scores)

        assert isinstance(labels, Tensor)
        expected_labels = in_labels[:1]
        assert labels.shape == expected_labels.shape
        assert labels.equal(expected_labels)

    def test_nms_below_iou_threshold(self):
        """
        Test that NMS does not filter boxes below the IoU threshold.
        """
        # Create dummy data
        device = torch.device("cpu")
        dtype = torch.float32
        in_boxes = torch.tensor(
            ([[0.05, 0.05, 0.1, 0.1], [0.05, 0.05, 0.05, 0.05]]),
            dtype=dtype,
            device=device,
        )
        in_scores = torch.tensor([1, 1], dtype=dtype, device=device)
        in_labels = torch.tensor([0, 0], dtype=torch.int, device=device)

        boxes, scores, labels = BoxUtils.nms(in_boxes, in_scores, in_labels, 0.26)

        assert isinstance(boxes, Tensor)
        assert boxes.shape == in_boxes.shape
        assert boxes.allclose(in_boxes)

        assert isinstance(scores, Tensor)
        assert scores.shape == in_scores.shape
        assert scores.allclose(in_scores)

        assert isinstance(labels, Tensor)
        assert labels.shape == in_labels.shape
        assert labels.equal(in_labels)

    def test_nms_empty_boxes(self):
        """
        Test NMS doesn't throw an error when an empty boxes tensor is passed in.
        """
        # Create dummy data
        device = torch.device("cpu")
        dtype = torch.float32
        in_boxes = torch.zeros((0, 4), dtype=dtype, device=device)
        in_scores = torch.zeros((0,), dtype=dtype, device=device)
        in_labels = torch.zeros((0,), dtype=torch.int, device=device)

        boxes, scores, labels = BoxUtils.nms(in_boxes, in_scores, in_labels, 0.26)

        assert boxes.allclose(in_boxes)
        assert scores.allclose(in_scores)
        assert labels.allclose(in_labels)
