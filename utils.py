from typing import Tuple, Union, Optional
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def fit_mask_to_bbox(mask: np.ndarray, bbox_size: Tuple[int, int]) -> np.ndarray:
    """
    Pads or crops the mask to the bounding box size.
    Args:
        mask: np.ndarray of shape (mask_height, mask_width)
        bbox_size: Tuple[int, int] of the bounding box size (height, width)
    Returns:
        np.ndarray of shape (height, width)
    """
    return np.pad(mask, [(0, max(0, bbox_size[0] - mask.shape[0])), (0, max(0, bbox_size[1] - mask.shape[1]))])[:bbox_size[0], :bbox_size[1]]


def normalized_bbox_to_pixel_coords(bbox, image_width, image_height):
    """
    Convert normalized bounding box [x, y, width, height] to pixel coordinates.
    
    Args:
        bbox: Normalized bounding box [x, y, width, height]
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        tuple: (x1, y1, x2, y2) pixel coordinates
    """
    x1 = int(bbox[0] * image_width)
    y1 = int(bbox[1] * image_height)
    x2 = int((bbox[0] + bbox[2]) * image_width)
    y2 = int((bbox[1] + bbox[3]) * image_height)
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return x1, y1, x2, y2


def box_iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a["bounding_box"]
    bx, by, bw, bh = box_b["bounding_box"]

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area

    if union == 0:
        return 0.0

    return inter_area / union


def evaluate(original_detections, propagated_detections):
    """
    Evaluate the propagation against the original detection.
    Args:
        original_detections: The original detections
        propagated_detections: The propagated detections
    Returns:
        float: The evaluation score
    """
    # TODO(neeraja): replace this with a more standard evaluation metric

    # for now, only evaluates bounding boxes
    # TODO(neeraja): implement for masks

    G = len(original_detections.detections)
    P = len(propagated_detections.detections)

    if min(G, P) == 0:
        return 0.0

    # IoU matrix: shape (G, P)
    iou_matrix = np.zeros((G, P), dtype=np.float32)
    for i, gt in enumerate(original_detections.detections):
        for j, pred in enumerate(propagated_detections.detections):
            iou_matrix[i, j] = box_iou(gt, pred)

    # Hungarian finds MIN cost → negate IoU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    total_iou = 0.0
    for i, j in zip(row_ind, col_ind):
        # area_p = propagated_detections[j]["bounding_box"][2] * propagated_detections[j]["bounding_box"][3]
        # area_gt = original_detections[i]["bounding_box"][2] * original_detections[i]["bounding_box"][3]
        total_iou += iou_matrix[i, j]

    # Unmatched predictions/ground_truths contribute IoU = 0 implicitly
    return total_iou / max(G, P)


def evaluate_matched(original_detections, propagated_detections):
    """
    Evaluate the propagation against the original detection.
    Args:
        original_detections: The original detections
        propagated_detections: The propagated detections
    Returns:
        float: The evaluation score
    """
    # TODO(neeraja): implement for masks
    if len(original_detections.detections) == 0:
        return 0.0
    
    assert len(original_detections.detections) == len(propagated_detections.detections)
    total_iou = 0.0
    for od, pd in zip(original_detections.detections, propagated_detections.detections):
        total_iou += box_iou(od, pd)
    return total_iou / len(original_detections.detections)
