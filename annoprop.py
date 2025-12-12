import numpy as np
import cv2
from typing import Tuple, Union

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
        # Get the mask from the source segmentation
        source_mask = detection.mask
        # TODO(neeraja): handle the bounding_box case

        # this is relative to the bbox
        source_bbox = detection.bounding_box
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(source_bbox, target_width, target_height)
        source_mask_fitted = fit_mask_to_bbox(source_mask, (y2-y1, x2-x1))

        # make it relative to the target frame
        source_mask_framed = np.zeros((target_height, target_width), bool)
        source_mask_framed[y1:y2, x1:x2] = source_mask_fitted

        source_mask_framed = source_mask_framed.astype(np.uint8)
        
        # Initialize GrabCut mask
        # GrabCut mask values:
        # 0 = definitely background
        # 1 = definitely foreground
        # 2 = probably background
        # 3 = probably foreground
        grabcut_mask = np.zeros((target_height, target_width), np.uint8)
        
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
        
        # Initialize GrabCut mask:
        # - Interior pixels: probably foreground (allows some refinement)
        # - Edge pixels: probably background (allows boundary refinement)
        # - Background pixels: probably background
        grabcut_mask[interior_mask > 0] = cv2.GC_PR_FGD  # interior = probably foreground
        grabcut_mask[edge_mask > 0] = cv2.GC_PR_BGD  # edges = probably background (allows refinement)
        grabcut_mask[binary_mask == 0] = cv2.GC_PR_BGD  # background = probably background
        
        # Optionally, mark high-confidence interior as definitely foreground
        # This can help preserve well-segmented regions
        # Uncomment below to enable:
        # if interior_mask.sum() > 100:  # Only if mask is large enough
        #     grabcut_mask[interior_mask > 0] = cv2.GC_FGD  # definitely foreground
        
        # Initialize background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
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
        refined_mask_fitted = refined_mask[y1:y2, x1:x2]
        
        # Create new segmentation with refined mask
        new_detection = fo.Detection(
            bounding_box=new_bbox,
            mask=refined_mask_fitted,
            label=detection.label if hasattr(detection, 'label') else None,
        )
        propagated_detections.append(new_detection)
    
    return fo.Detections(detections=propagated_detections)


def propagate_annotations(
    view: Union[fo.core.dataset.Dataset, fo.core.view.DatasetView],
	exemplar_frame_field: str, 
	annotation_field: str,
	exemplar_assignments: dict,
) -> None:
    """
    Propagate annotations from exemplar frames to all the frames.
    Args:
        view: The view to propagate annotations from
        exemplar_frame_field: The field name of the exemplar frame
        annotation_field: The field name of the annotation
        exemplar_assignments: {sample_id: [exemplar_frame_ids]} for each sample in the view
    """
    exemplar_detections = {}
    for sample in view:
        if sample[exemplar_frame_field]:
            exemplar_detections[sample.id] = sample[annotation_field]
    
    for sample in view:
        if not sample[exemplar_frame_field] and (sample.id in exemplar_assignments) and (len(exemplar_assignments[sample.id]) > 0):
            exemplar_frame_ids = exemplar_assignments[sample.id]
            # TODO(neeraja): handle multiple exemplar frames for the same sample
            exemplar_detection = exemplar_detections[exemplar_frame_ids[0]]

            sample_frame = cv2.imread(sample.filepath)
            propagated_detection = propagate_detections_with_grabcut(sample_frame, exemplar_detection)
            sample[annotation_field] = propagated_detection
            sample.save()
    
    return None