import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

from ssd.structs import FrameLabels


class BoxUtils:
    @staticmethod
    def regression_domain_to_image_domain(
        box_regressions: Tensor, anchors: Tensor
    ) -> Tensor:
        """
        Convert the box component of the network head's output into the image domain.
        Specifically, the network regresses `dx`, `dy`, `dw` and `dh` which adjust the
        anchor boxes instead of directly regressing the output boxes. One can calculate
        the regressed boxes using the following functions:

            ```python
            cx = dx * anchor_w + anchor_cx
            cy = dy * anchor_h + anchor_cy
            w = exp(dw) * anchor_w
            h = exp(dh) * anchor_h
            ```

        Paramaters
        ----------
        box_regressions:
            The box component produced by the network's head. This should have a shape
            of `(batch_size, num_anchor_boxes, 4)`. The last dimension should be
            structured as `(dx, dy, dw, dh)`.

        anchors:
            The anchor boxes for the network. This should have a shape of `(batch_size,
            num_anchor_boxes, 4)`. The last dimensions should be structured as `(cx, cy,
            w, h)`.

        Returns
        -------
        predicted_boxes:
            The predicted boxes in the image's coordinate system. This will have a shape
            of `(batch_size, num_anchor_boxes, 4)`. With the last dimensions being
            structured as `(cx, cy, w, h)`.
        """

        # Extract the "deltas" predicted by the model
        dx = box_regressions[..., 0]
        dy = box_regressions[..., 1]
        dw = box_regressions[..., 2]
        dh = box_regressions[..., 3]

        # Extract the components of the anchor boxes
        anchor_cxs = anchors[..., 0]
        anchor_cys = anchors[..., 1]
        anchor_ws = anchors[..., 2]
        anchor_hs = anchors[..., 3]

        # Calculate the predicted boxes in the image's coordinate system
        pred_cx = dx * anchor_ws + anchor_cxs
        pred_cy = dy * anchor_hs + anchor_cys
        pred_w = torch.exp(dw) * anchor_ws
        pred_h = torch.exp(dh) * anchor_hs

        return torch.stack((pred_cx, pred_cy, pred_w, pred_h), dim=-1)

    @staticmethod
    def image_domain_to_regression_domain(boxes: Tensor, anchors: Tensor) -> Tensor:
        """
        Convert boxes in the image domain (usually ground truth boxes) into the
        regression domain. One can convert boxes from the image domain into the
        regression domain using:

            ```python
            dx = (cx - anchor_cx) / anchor_w
            dy = (cy - anchor_cy) / anchor_h
            dw = ln(w / anchor_w)
            dh = ln(h / anchor_h)
            ```

        Paramaters
        ----------
        boxes:
            A tensor of boxes in the image domain. This should have a shape of
            `(batch_size, num_objects, 4)`. The last dimension should be structured as
            `(cx, cy, w, h)`.

        anchors:
            The anchor boxes that are closest to the input boxes. This should have a
            shape of `(batch_size, num_objects, 4)`. The last dimensions should be
            structured as `(cx, cy, w, h)`.

        Returns
        -------
        regression_boxes:
            The boxes converted to the regression domain. This will have a shape of
            `(batch_size, num_objects, 4)`. With the last dimensions being structured as
            `(dx, dy, dw, dh)`.
        """

        # Extract the box components
        cx = boxes[..., 0]
        cy = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]

        # Extract the anchor box components
        anchor_cx = anchors[..., 0]
        anchor_cy = anchors[..., 1]
        anchor_w = anchors[..., 2]
        anchor_h = anchors[..., 3]

        # Calculate deltas in the regression domain
        dx = (cx - anchor_cx) / anchor_w
        dy = (cy - anchor_cy) / anchor_h
        dw = torch.log(w / anchor_w)
        dh = torch.log(h / anchor_h)

        return torch.stack((dx, dy, dw, dh), dim=-1)

    @staticmethod
    def find_indices_of_best_anchor_boxes(
        anchors: Tensor, ground_truth_boxes: list[Tensor]
    ) -> list[Tensor]:
        """
        Finds the indices of anchor boxes with the highest IoU to the ground truth
        boxes.

        Parameters
        ----------
        anchors:
            The anchor boxes for the network. This should have a shape of `(batch_size,
            num_anchor_boxes, 4)`. The last dimensions should be structured as `(cx, cy,
            w, h)`. The anchor boxes should be in the image's domain.

        ground_truth_boxes:
            The ground truth boxes associated with each image. These boxes are in the
            image's domain. The number of elements in the list is equal to `batch_size`
            and the size of the tensor is `(num_objects, 4)`. With the last dimension
            being structured as `(cx, cy, w, h)`.

        Returns
        -------
        best_anchor_indicies:
            The indicies of the anchor boxes that best match the ground truth boxes.
            The number of elements in the list is `batch_size` with the tensor having
            shape `(num_objects,)`.
        """
        # Check the batch size of the anchors matches that of the ground truth boxes
        if anchors.shape[0] != len(ground_truth_boxes):
            raise ValueError(
                f"Batch size mismatch: ABS={anchors.shape[0]} "
                f"GTBS={len(ground_truth_boxes)}."
            )

        # Determine which anchor boxes have the highest IoU with the labels
        best_anchor_indices: list[Tensor] = []
        batch_size = len(ground_truth_boxes)
        for idx in range(batch_size):
            # Calculate anchor box IoU
            image_anchors = anchors[idx, ...]
            image_ground_truth_boxes = ground_truth_boxes[idx]
            image_anchors_xyxy = box_convert(image_anchors, "cxcywh", "xyxy")
            image_gt_xyxy = box_convert(image_ground_truth_boxes, "cxcywh", "xyxy")
            iou_matrix = box_iou(image_anchors_xyxy, image_gt_xyxy)

            # Find which anchor box best fits the label
            best_anchor_indices.append(iou_matrix.max(dim=0).indices)

        return best_anchor_indices
