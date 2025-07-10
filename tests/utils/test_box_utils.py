import torch
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
        # Construct dummy inputs
        anchors = torch.tensor([[[10, 10, 6, 6], [100, 100, 60, 60]]])
        labels = [torch.tensor([[88, 120, 40, 40]])]

        # Try to find the best anchor box indices
        best_anchor_indices = BoxUtils.find_indices_of_best_anchor_boxes(
            anchors, labels
        )

        # Check the output is correct
        assert isinstance(best_anchor_indices, list)
        assert len(best_anchor_indices) == 1
        assert best_anchor_indices[0] == torch.tensor([1])

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
        best_anchor_indices = BoxUtils.find_indices_of_best_anchor_boxes(
            anchors, ground_truth
        )

        # Check the output is correct
        assert isinstance(best_anchor_indices, list)
        assert len(best_anchor_indices) == batch_size
        assert best_anchor_indices[0].shape == (num_objects,)
