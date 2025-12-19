from typing import Tuple, Union, Optional
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

import fiftyone as fo


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
    return (x1, y1, x2, y2)


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


def propagate_detections_no_op(target_frame, source_detections):
    """
    Propagate detections as-is (Do nothing).
    """
    return source_detections


def propagate_detections_with_grabcut(target_frame, source_detections):
    """
    Propagate detections from source_detections to target_frame using cv2's grabcut.
    
    Args:
        target_frame: The target frame to propagate to
        source_detections: The detections from source_frame (fo.Detections)

    Returns:
        fo.Detections: New detections for the target frame
    """
    # HYPERPARAMS
    EDGE_KERNEL_SIZE = 3  # larger = thicker border region marked as uncertain
    EDGE_ITERATIONS = 1
    GRABCUT_ITERATIONS = 5

    target_height, target_width = target_frame.shape[:2]
    
    propagated_detections = []
    
    for detection in source_detections.detections:
        # Initialize GrabCut mask
        # GrabCut mask values:
        # 0 = definitely background
        # 1 = definitely foreground
        # 2 = probably background
        # 3 = probably foreground
        grabcut_mask = np.zeros((target_height, target_width), np.uint8)
        
        # Initialize background and foreground models for each detection
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Get bounding box from source detection
        source_bbox = detection.bounding_box
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(
            source_bbox, target_width, target_height
        )
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, target_width - 1))
        y1 = max(0, min(y1, target_height - 1))
        x2 = max(x1 + 1, min(x2, target_width))
        y2 = max(y1 + 1, min(y2, target_height))
        
        # Get the mask from the source segmentation
        source_mask = detection.mask

        if source_mask is not None:
            source_mask_fitted = fit_mask_to_bbox(source_mask, (y2-y1, x2-x1))
            # make it relative to the target frame
            source_mask_framed = np.zeros((target_height, target_width), bool)
            source_mask_framed[y1:y2, x1:x2] = source_mask_fitted
            source_mask_framed = source_mask_framed.astype(np.uint8)
        
            # Convert source mask to GrabCut mask format
            # Assume source mask: 0 = background, >0 = foreground
            binary_mask = (source_mask_framed > 0).astype(np.uint8)
            
            # Find mask edges using morphological erosion
            # Erode the mask to get the interior (removes border pixels)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_KERNEL_SIZE, EDGE_KERNEL_SIZE))
            eroded_mask = cv2.erode(binary_mask, kernel, iterations=EDGE_ITERATIONS)
            
            # Edge pixels are those in the original mask but not in the eroded mask
            edge_mask = binary_mask - eroded_mask
            
            # Interior pixels (high confidence foreground)
            interior_mask = eroded_mask
            
            # Refine GrabCut mask:
            # - Interior pixels: probably foreground (allows some refinement)
            # - Edge pixels: probably background (allows boundary refinement)
            # - Background pixels: probably background
            grabcut_mask[interior_mask > 0] = cv2.GC_PR_FGD  # interior = probably foreground
            grabcut_mask[edge_mask > 0] = cv2.GC_PR_BGD  # edges = probably background (allows refinement)
            grabcut_mask[binary_mask == 0] = cv2.GC_PR_BGD  # background = probably background

            # Apply grabcut with mask initialization
            cv2.grabCut(
                target_frame,
                grabcut_mask,
                None,  # No rectangle needed when using mask
                bgd_model,
                fgd_model,
                GRABCUT_ITERATIONS,
                cv2.GC_INIT_WITH_MASK
            )

            # Create binary mask: definitely/probably foreground (1, 3) become foreground
            refined_mask = np.where((grabcut_mask == cv2.GC_FGD) | 
                               (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            
        else:
            # Create rectangle for grabcut (x, y, width, height)
            rect = (x1, y1, x2 - x1, y2 - y1)
            # Apply grabcut
            cv2.grabCut(
                target_frame,
                grabcut_mask,
                rect,
                bgd_model,
                fgd_model,
                5,  # number of iterations
                cv2.GC_INIT_WITH_RECT
            )
            # Create binary mask: 2 and 3 are foreground
            refined_mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')
        
        # find its new bbox
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            new_bbox = [x/target_width, y/target_height, w/target_width, h/target_height]
        else:
            print("Warning: No contour found for segmentation")
            continue

        # make it relative to the new bbox
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(new_bbox, target_width, target_height)
        
        # Create new segmentation with refined mask
        new_detection = fo.Detection(
            bounding_box=new_bbox,
            mask=refined_mask[y1:y2, x1:x2] if source_mask is not None else None,
            label=detection.label if hasattr(detection, 'label') else None,
        )
        propagated_detections.append(new_detection)
    
    return fo.Detections(detections=propagated_detections)


def propagate_annotations(
    view: Union[fo.core.dataset.Dataset, fo.core.view.DatasetView],
	exemplar_frame_field: str,
	input_annotation_field: str,
	output_annotation_field: str,
	exemplar_assignments: dict,
    evaluate_propagation: Optional[bool] = True,
) -> None:
    """
    Propagate annotations from exemplar frames to all the frames.
    Args:
        view: The view to propagate annotations from
        exemplar_frame_field: The field name of the exemplar frame
                              TODO(neeraja): Explore whether we can remove this field
        annotation_field: The field name of the annotation to copy from the exemplar frame
        output_annotation_field: The field name of the annotation to save to the target frame
        exemplar_assignments: {sample_id: [exemplar_frame_ids]} for each sample in the view
        evaluate_propagation: Whether to evaluate the propagation against
                              the input annotation field present in the propagation targets.
    """
    score = 0.0

    for sample in view:
        if sample[exemplar_frame_field]:
            sample[output_annotation_field] = sample[input_annotation_field]
        elif (sample.id in exemplar_assignments) and (len(exemplar_assignments[sample.id]) > 0):
            exemplar_frame_ids = exemplar_assignments[sample.id]

            # TODO(neeraja): handle multiple exemplar frames for the same sample
            exemplar_frame = view[exemplar_frame_ids[0]]
            exemplar_detections = exemplar_frame[input_annotation_field]

            sample_frame = cv2.imread(sample.filepath)
            # propagated_detections = propagate_detections_no_op(sample_frame, exemplar_detections)
            propagated_detections = propagate_detections_with_grabcut(sample_frame, exemplar_detections)
            sample[output_annotation_field] = propagated_detections
            sample.save()

            # If the sample already has an input annotation field, evaluate against it
            if evaluate_propagation and sample[input_annotation_field]:
                original_detections = sample[input_annotation_field]
                sample_score = evaluate(original_detections, propagated_detections)
                print(f"Sample {sample.id} score: {sample_score}")
                score += sample_score
    
    return score
