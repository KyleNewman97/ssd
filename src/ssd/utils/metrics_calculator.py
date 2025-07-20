import torch
import numpy as np
from torch import Tensor
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm

from ssd.structs import FrameDetections, FrameLabels
from ssd.utils.meta_logger import MetaLogger


class MetricsCalculator(MetaLogger):
    def __init__(
        self,
        num_classes: int,
        iou_thresholds: np.ndarray | list[float] = np.linspace(0.5, 0.95, 10),
        confidence_thresholds: np.ndarray | list[float] = np.linspace(0.1, 1, 10),
    ):
        MetaLogger.__init__(self)

        self._num_classes = num_classes
        self._iou_thresholds = iou_thresholds
        self._confidence_thresholds = confidence_thresholds

        self._frame_per_class_ious: list[dict[int, Tensor]] = []
        self._frame_per_class_scores: list[dict[int, Tensor]] = []
        self._frame_per_class_label_count: list[dict[int, int]] = []

        self._tps: Tensor | None = None
        self._fps: Tensor | None = None
        self._fns: Tensor | None = None

    def update(
        self, frame_detections: list[FrameDetections], frame_labels: list[FrameLabels]
    ):
        # On update ensure TPs, FPs and FNs are marked as out of date
        self._tps, self._fps, self._fns = None, None, None

        for detections, labels in zip(frame_detections, frame_labels):
            detection_xyxy_boxes = box_convert(detections.boxes, "cxcywh", "xyxy")
            label_xyxy_boxes = box_convert(labels.boxes, "cxcywh", "xyxy")

            class_ids: Tensor = detections.class_ids.unique()
            self._frame_per_class_ious.append({})
            self._frame_per_class_scores.append({})
            for class_id in class_ids:
                # Calculate class masks
                det_class_mask = detections.class_ids == class_id
                lab_class_mask = labels.class_ids == class_id

                # Extract boxes and scores
                det_boxes = detection_xyxy_boxes[det_class_mask, :]
                det_scores = detections.scores[det_class_mask]
                lab_boxes = label_xyxy_boxes[lab_class_mask, :]

                # Calculate IoU for each box: N dets, M labs produces (N, M) matrix
                iou_matrix = box_iou(det_boxes, lab_boxes)

                # Add the classes box IoUs and scores
                cls_id = int(class_id.item())
                self._frame_per_class_ious[-1][cls_id] = iou_matrix
                self._frame_per_class_scores[-1][cls_id] = det_scores

            # Update the number of objects per frame per class
            label_class_ids: Tensor = labels.class_ids.unique()
            self._frame_per_class_label_count.append({})
            for class_id in label_class_ids:
                cls_id = int(class_id.item())
                self._frame_per_class_label_count[-1][cls_id] = int(
                    (labels.class_ids == class_id).sum().item()
                )

    def tps(self) -> Tensor:
        """
        Calculate the TPs on a per-class basis at various confidence and IoU thresholds.

        Returns
        -------
        tps:
            A tensor shaped as `(num_confs, num_ious, num_classes)` with each entry
            representing the corresponding `(conf, iou, class)` tuple's TP count.
        """
        if self._tps is not None:
            return self._tps

        num_confs = len(self._confidence_thresholds)
        num_ious = len(self._iou_thresholds)
        self._tps = torch.zeros(
            (
                num_confs,
                num_ious,
                self._num_classes,
            ),
            dtype=torch.int,
        )

        num_frames = len(self._frame_per_class_ious)
        frame_iterator = zip(self._frame_per_class_ious, self._frame_per_class_scores)
        frame_iterator = tqdm(frame_iterator, ncols=88, total=num_frames)
        # Loop over frames
        for per_class_ious, per_class_scores in frame_iterator:
            # Loop over classes observed in the frame
            for class_id in per_class_ious:
                # Find the IoU matrix and scores for specified class in this frame
                iou_matrix = per_class_ious[class_id]
                scores = per_class_scores[class_id]

                for conf_idx, conf_threshold in enumerate(self._confidence_thresholds):
                    # Set the IoU for all detections below the confidence threshold to 0
                    iou_matrix_masked = iou_matrix.clone()
                    score_mask = scores <= float(conf_threshold)
                    iou_matrix_masked[score_mask, :] = 0

                    for iou_idx, iou_threshold in enumerate(self._iou_thresholds):
                        det_idxs, lab_idxs = torch.where(
                            iou_matrix_masked >= float(iou_threshold)
                        )
                        iou_values = iou_matrix_masked[det_idxs, lab_idxs]

                        # Sort the det and lab indicies by IoU (descending)
                        sorted_idxs = torch.argsort(iou_values, descending=True)
                        det_idxs = det_idxs[sorted_idxs]
                        lab_idxs = lab_idxs[sorted_idxs]

                        # Find detection and label pairs
                        tps = 0
                        matched_dets, matched_labs = set(), set()
                        idx_iterator = zip(
                            det_idxs.cpu().tolist(), lab_idxs.cpu().tolist()
                        )
                        for det_idx, lab_idx in idx_iterator:
                            if det_idx in matched_dets or lab_idx in matched_labs:
                                continue
                            matched_dets.add(det_idx)
                            matched_labs.add(lab_idx)
                            tps += 1

                        self._tps[conf_idx, iou_idx, class_id] += tps

        return self._tps

    def fps(self) -> Tensor:
        """
        Calculate the FPs on a per-class basis at various confidence and IoU thresholds.

        Returns
        -------
        fps:
            A tensor shaped as `(num_confs, num_ious, num_classes)` with each entry
            representing the corresponding `(conf, iou, class)` tuple's FP count.
        """
        if self._fps is not None:
            return self._fps

        num_confs = len(self._confidence_thresholds)
        num_ious = len(self._iou_thresholds)
        self._fps = torch.zeros(
            (
                num_confs,
                num_ious,
                self._num_classes,
            ),
            dtype=torch.int,
        )

        # Calculate the number of true positives
        tps = self.tps()

        # We can calculate the FPs using the equation FPs = NUM_DETS - TPs
        for per_class_scores in self._frame_per_class_scores:
            for class_id in per_class_scores:
                scores = per_class_scores[class_id]
                for conf_idx, conf_threshold in enumerate(self._confidence_thresholds):
                    num_dets = (scores >= conf_threshold).sum()
                    self._fps[conf_idx, :, class_id] += (
                        num_dets - tps[conf_idx, :, class_id]
                    )

        return self._fps

    def fns(self) -> Tensor:
        """
        Calculate the FNs on a per-class basis at various confidence and IoU thresholds.

        Returns
        -------
        fns:
            A tensor shaped as `(num_confs, num_ious, num_classes)` with each entry
            representing the corresponding `(conf, iou, class)` tuple's FN count.
        """
        if self._fns is not None:
            return self._fns

        num_confs = len(self._confidence_thresholds)
        num_ious = len(self._iou_thresholds)
        self._fns = torch.zeros(
            (
                num_confs,
                num_ious,
                self._num_classes,
            ),
            dtype=torch.int,
        )

        # Calculate the number of true positives
        tps = self.tps()

        # We can calculate the FNs using the equation FNs = NUM_LABELS - TPs
        for per_class_label_count in self._frame_per_class_label_count:
            for class_id in per_class_label_count:
                count = per_class_label_count[class_id]
                self._fns[:, :, class_id] += count - tps[:, :, class_id]

        return self._fns
