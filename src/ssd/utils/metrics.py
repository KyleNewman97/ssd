from torch import Tensor
from torchvision.ops import box_convert, box_iou

from ssd.structs import FrameDetections, FrameLabels


class Metrics:
    def __init__(self):
        self._frame_detections: list[FrameDetections] = []
        self._frame_labels: list[FrameLabels] = []

    def update(
        self, frame_detections: list[FrameDetections], frame_labels: list[FrameLabels]
    ):
        self._frame_detections += frame_detections
        self._frame_labels += frame_labels

    @staticmethod
    def frame_true_positives(
        detections: FrameDetections,
        labels: FrameLabels,
        num_classes: int,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> dict[int, int]:
        """
        Calculates the number of true positives on a single frame.
        """
        class_id_to_tps = {i: 0 for i in range(num_classes)}

        # Filter out detections below the confidence threshold
        detection_boxes = detections.boxes[detections.scores >= confidence_threshold, :]

        if detection_boxes.shape[0] == 0:
            return class_id_to_tps

        detection_xyxy_boxes = box_convert(detection_boxes, "cxcywh", "xyxy")
        label_xyxy_boxes = box_convert(labels.boxes, "cxcywh", "xyxy")

        class_ids: Tensor = detections.class_ids.unique()
        for class_id in class_ids:
            det_boxes = detection_xyxy_boxes[detections.class_ids == class_id, :]
            lab_boxes = label_xyxy_boxes[labels.class_ids == class_id, :]
            iou_matrix = box_iou(det_boxes, lab_boxes)

            # Iterate through the calculated IoUs
            # Each loop find the detection label pair with the highest IoU.
            #   - If this IoU is below the threshold then we stop - no more pairs will
            #       be above the threshold.
            #   - If the IoU is above the threshold count another true positive and then
            #       set the IoUs for the detection and label boxes to 0 so they cannot
            #       be matched again.
            # We perform this loop a maximum of `min(num_dets, num_labs)` times, because
            # after this point all values will be zero within the matrix.
            true_positives = 0
            for _ in range(min(iou_matrix.shape)):
                max_iou = iou_matrix.max()
                if max_iou < iou_threshold:
                    break

                true_positives += 1

                max_idx = iou_matrix.argmax()
                max_row, max_col = divmod(max_idx.item(), iou_matrix.shape[1])
                iou_matrix[int(max_row), :] = 0
                iou_matrix[:, int(max_col)] = 0

            class_id_to_tps[class_id.item()] = true_positives  # type: ignore

        return class_id_to_tps

    @staticmethod
    def frame_false_positives(
        detections: FrameDetections,
        labels: FrameLabels,
        num_classes: int,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> dict[int, int]:
        """
        Calculates the number of false positives in a single frame.
        """
        # Initially assume all detections are false positives
        class_id_to_fps = {
            i: int((detections.class_ids == i).sum().item()) for i in range(num_classes)
        }

        # Calculate the true positives
        class_id_to_tps = Metrics.frame_true_positives(
            detections, labels, num_classes, confidence_threshold, iou_threshold
        )

        # Adjust the false positives based on the number of true positives observed
        for class_id in range(num_classes):
            class_id_to_fps[class_id] -= class_id_to_tps[class_id]

        return class_id_to_fps

    @staticmethod
    def frame_false_negatives(
        detections: FrameDetections,
        labels: FrameLabels,
        num_classes: int,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> dict[int, int]:
        """
        Calculates the number of false negatives in a single frame.
        """
        # Initially assume all labels are false negatives
        class_id_to_fns = {
            i: int((labels.class_ids == i).sum().item()) for i in range(num_classes)
        }

        # Calculate the true positives
        class_id_to_tps = Metrics.frame_true_positives(
            detections, labels, num_classes, confidence_threshold, iou_threshold
        )

        # Adjust the false negatives based on the number of true positives observed
        for class_id in range(num_classes):
            class_id_to_fns[class_id] -= class_id_to_tps[class_id]

        return class_id_to_fns

    def per_class_precision(
        self, confidence: float, iou_threshold: float
    ) -> dict[int, float]:
        pass

    def mean_precision(self, confidence: float, iou_threshold: float) -> float:
        pass
