import fiftyone as fo
import cv2
import numpy as np
from PIL import Image
from typing import Tuple

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

from eta.core.video import VideoProcessor


TEST_FREQUENCY = 20


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



def create_detection_from_bbox(detection, bbox):
    """
    Create a new Detection object from an existing detection with a new bounding box.
    
    Args:
        detection: Source fo.Detection object
        bbox: New bounding box [x, y, width, height] in normalized coordinates
        
    Returns:
        fo.Detection: New detection with updated bounding box
    """
    return fo.Detection(
        label=detection.label,
        bounding_box=bbox,
        confidence=detection.confidence if hasattr(detection, 'confidence') else None,
    )


def propagate_bboxes_with_grabcut(source_frame, target_frame, source_detections):
    """
    Propagate detections from source_frame to target_frame using cv2's grabcut.
    
    Args:
        source_frame: The source frame with detections
        target_frame: The target frame to propagate to
        source_detections: The detections from source_frame
        
    Returns:
        fo.Detections: New detections for the target frame
    """
    target_height, target_width = target_frame.shape[:2]
    
    propagated_detections = []
    
    for detection in source_detections.detections:
        # Create a fresh mask for each detection
        mask = np.zeros((target_height, target_width), np.uint8)
        
        # Initialize background and foreground models for each detection
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Get bounding box from source detection
        bbox = detection.bounding_box
        
        # Convert normalized bbox to pixel coordinates
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(
            bbox, target_width, target_height
        )
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, target_width - 1))
        y1 = max(0, min(y1, target_height - 1))
        x2 = max(x1 + 1, min(x2, target_width))
        y2 = max(y1 + 1, min(y2, target_height))
        
        # Create rectangle for grabcut (x, y, width, height)
        rect = (x1, y1, x2 - x1, y2 - y1)
        
        # Apply grabcut
        cv2.grabCut(
            target_frame,
            mask,
            rect,
            bgd_model,
            fgd_model,
            5,  # number of iterations
            cv2.GC_INIT_WITH_RECT
        )
        
        # Create binary mask: 2 and 3 are foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Find bounding box of the segmented region
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Convert back to normalized coordinates
            norm_x = x / target_width
            norm_y = y / target_height
            norm_w = w / target_width
            norm_h = h / target_height
            
            # Create new detection with propagated bounding box
            new_bbox = [norm_x, norm_y, norm_w, norm_h]
            new_detection = create_detection_from_bbox(detection, new_bbox)
            propagated_detections.append(new_detection)
        else:
            # If no contour found, use original bounding box
            new_detection = create_detection_from_bbox(detection, bbox)
            propagated_detections.append(new_detection)
    
    return fo.Detections(detections=propagated_detections)


def propagate_bboxes_with_densecrf(source_frame, target_frame, source_detections):
    """
    Propagate detections from source_frame to target_frame using DenseCRF.
    
    Args:
        source_frame: The source frame with detections
        target_frame: The target frame to propagate to
        source_detections: The detections from source_frame
        
    Returns:
        fo.Detections: New detections for the target frame
    """
    target_height, target_width = target_frame.shape[:2]
    n_points = target_width * target_height
    
    propagated_detections = []
    
    for detection in source_detections.detections:
        # Get bounding box from source detection
        bbox = detection.bounding_box
        
        # Convert normalized bbox to pixel coordinates
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(
            bbox, target_width, target_height
        )
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, target_width - 1))
        y1 = max(0, min(y1, target_height - 1))
        x2 = max(x1 + 1, min(x2, target_width))
        y2 = max(y1 + 1, min(y2, target_height))
        
        # Create unary potential: initialize based on bounding box
        # Higher probability inside bbox, lower outside
        unary = np.zeros((2, n_points), dtype=np.float32)  # 2 classes: bg, fg
        
        # Create a soft initialization: pixels inside bbox have higher foreground probability
        for y in range(target_height):
            for x in range(target_width):
                idx = y * target_width + x
                if x1 <= x < x2 and y1 <= y < y2:
                    # Inside bbox: higher foreground probability
                    unary[0, idx] = 0.3  # background
                    unary[1, idx] = 0.7  # foreground
                else:
                    # Outside bbox: higher background probability
                    unary[0, idx] = 0.9  # background
                    unary[1, idx] = 0.1  # foreground
        
        # Normalize to probabilities
        unary = -np.log(unary + 1e-8)  # Convert to negative log probabilities
        
        # Create DenseCRF
        d = dcrf.DenseCRF2D(target_width, target_height, 2)
        
        # Set unary potential
        d.setUnaryEnergy(unary)
        
        # Add pairwise Gaussian potential (spatial smoothness)
        d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL, 
                             normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        # Add pairwise bilateral potential (color-dependent)
        # Convert BGR to RGB for pydensecrf
        rgb_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=rgb_frame, compat=10,
                              kernel=dcrf.DIAG_KERNEL, 
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        # Run inference
        Q = d.inference(5)  # 5 iterations
        
        # Get the MAP prediction (foreground class = 1)
        map_result = np.argmax(Q, axis=0).reshape((target_height, target_width))
        
        # Create binary mask
        mask = (map_result == 1).astype('uint8')
        
        # Find bounding box of the segmented region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Convert back to normalized coordinates
            norm_x = x / target_width
            norm_y = y / target_height
            norm_w = w / target_width
            norm_h = h / target_height
            
            # Create new detection with propagated bounding box
            new_bbox = [norm_x, norm_y, norm_w, norm_h]
            new_detection = create_detection_from_bbox(detection, new_bbox)
            propagated_detections.append(new_detection)
        else:
            # If no contour found, use original bounding box
            new_detection = create_detection_from_bbox(detection, bbox)
            propagated_detections.append(new_detection)
    
    return fo.Detections(detections=propagated_detections)


def propagate_segmentations_with_grabcut(source_frame, target_frame, source_segmentations):
    """
    Propagate segmentations from source_frame to target_frame using cv2's grabcut.
    
    Args:
        source_frame: The source frame with segmentations (used for reference, not directly used)
        target_frame: The target frame to propagate to
        source_segmentations: The segmentations from source_frame (fo.Detections)
        
    Returns:
        fo.Detections: New segmentations for the target frame
    """
    # HYPERPARAMS
    EDGE_KERNEL_SIZE = 3  # larger = thicker border region marked as uncertain
    EDGE_ITERATIONS = 1
    GRABCUT_ITERATIONS = 5

    target_height, target_width = target_frame.shape[:2]
    source_height, source_width = source_frame.shape[:2]
    
    propagated_segmentations = []
    
    for segmentation in source_segmentations.detections:
        # Get the mask from the source segmentation
        source_mask = segmentation.mask

        # this is relative to the bbox
        source_bbox = segmentation.bounding_box
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(source_bbox, source_width, source_height)
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
        new_segmentation = fo.Detection(
            bounding_box=new_bbox,
            mask=refined_mask_fitted,
            label=segmentation.label if hasattr(segmentation, 'label') else None,
        )
        propagated_segmentations.append(new_segmentation)
    
    return fo.Detections(detections=propagated_segmentations)


def propagate_segmentations_with_densecrf(source_frame, target_frame, source_segmentations):
    """
    Propagate segmentations from source_frame to target_frame using DenseCRF.
    
    Args:
        source_frame: The source frame with segmentations (used for reference, not directly used)
        target_frame: The target frame to propagate to
        source_segmentations: The segmentations from source_frame (fo.Detections)
        
    Returns:
        fo.Detections: New detections with propagated masks for the target frame
    """
    # HYPERPARAMS
    DENSECRF_SPATIAL_SMOOTHNESS = 5
    DENSECRF_COMPAT = 5
    DENSECRF_BILATERAL_SPATIAL_SMOOTHNESS = 40
    DENSECRF_BILATERAL_COLOR_SMOOTHNESS = 70
    DENSECRF_BILATERAL_COMPAT = 50
    UNARY_TEMPERATURE = 5.0
    DENSECRF_ITERATIONS = 10
    EPSILON = 1e-8

    target_height, target_width = target_frame.shape[:2]
    source_height, source_width = source_frame.shape[:2]
    n_points = target_width * target_height
    
    propagated_segmentations = []
    
    for segmentation in source_segmentations.detections:
        # Get the mask from the source segmentation (relative to bbox)
        source_mask = segmentation.mask
        
        # Get source bbox and convert to pixel coordinates
        source_bbox = segmentation.bounding_box
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(source_bbox, source_width, source_height)
        
        # Fit mask to bbox size
        source_mask_fitted = fit_mask_to_bbox(source_mask, (y2 - y1, x2 - x1))
        
        # Place mask in full frame coordinates (source frame)
        source_mask_framed = np.zeros((source_height, source_width), dtype=np.float32)
        source_mask_framed[y1:y2, x1:x2] = source_mask_fitted.astype(np.float32)
        
        # Resize to target frame dimensions if needed
        if (source_height != target_height) or (source_width != target_width):
            source_mask_framed = cv2.resize(source_mask_framed, (target_width, target_height), 
                                          interpolation=cv2.INTER_NEAREST)
        
        # Normalize mask to 0-1 range if needed
        if source_mask_framed.max() > 1.0:
            source_mask_framed = source_mask_framed / 255.0
        
        # Create unary potential from mask
        # Higher probability for foreground pixels, lower for background
        unary = np.zeros((2, n_points), dtype=np.float32)  # 2 classes: bg, fg
        
        # Convert mask to probabilities
        # Pixels with mask > 0.5 are likely foreground, others are likely background
        mask_flat = source_mask_framed.flatten()
        
        # Set probabilities based on mask values
        # Use soft probabilities: mask value directly influences foreground probability
        unary[0, :] = 1.0 - mask_flat  # background probability
        unary[1, :] = mask_flat  # foreground probability
        
        # Apply temperature to soften/harden unary confidence
        # Temperature > 1.0: softens (less confident, allows more adaptation)
        # Temperature < 1.0: sharpens (more confident, sticks closer to seed)
        if UNARY_TEMPERATURE != 1.0:
            unary = np.power(unary, 1.0 / UNARY_TEMPERATURE)
        
        # Add small epsilon to avoid log(0)
        unary = np.clip(unary, EPSILON, 1.0)
        
        # Normalize
        unary_sum = unary.sum(axis=0, keepdims=True)
        unary = unary / (unary_sum + EPSILON)
        
        # Convert to negative log probabilities
        unary = -np.log(unary)
        
        # Create DenseCRF
        d = dcrf.DenseCRF2D(target_width, target_height, 2)
        
        # Set unary potential
        d.setUnaryEnergy(unary)
        
        # Add pairwise Gaussian potential (spatial smoothness)
        d.addPairwiseGaussian(sxy=DENSECRF_SPATIAL_SMOOTHNESS, compat=DENSECRF_COMPAT)
        
        # Add pairwise bilateral potential (color-dependent)
        # Convert BGR to RGB for pydensecrf
        rgb_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
        d.addPairwiseBilateral(sxy=DENSECRF_BILATERAL_SPATIAL_SMOOTHNESS, srgb=DENSECRF_BILATERAL_COLOR_SMOOTHNESS, rgbim=rgb_frame, compat=DENSECRF_BILATERAL_COMPAT)
        
        # Run inference
        Q = d.inference(DENSECRF_ITERATIONS)
        
        # Get the MAP prediction (foreground class = 1)
        map_result = np.argmax(Q, axis=0).reshape((target_height, target_width))
        
        # Create binary mask
        refined_mask = (map_result == 1).astype(np.uint8) * 255
        
        # Find new bbox from refined mask
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            new_bbox = [x / target_width, y / target_height, w / target_width, h / target_height]
        else:
            print("Warning: No contour found for segmentation")
            new_bbox = segmentation.bounding_box
        
        # Extract mask relative to new bbox
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(new_bbox, target_width, target_height)
        refined_mask_fitted = refined_mask[y1:y2, x1:x2]
        
        # Create new detection with refined mask
        new_segmentation = fo.Detection(
            bounding_box=new_bbox,
            mask=refined_mask_fitted,
            label=segmentation.label if hasattr(segmentation, 'label') else None,
        )
        propagated_segmentations.append(new_segmentation)
    
    return fo.Detections(detections=propagated_segmentations)


def draw_bboxes_on_frame(frame, detections, color, thickness=2):
    """
    Draw detections on a frame.
    
    Args:
        frame: OpenCV image (numpy array)
        detections: fo.Detections object
        color: BGR color tuple (e.g., (0, 255, 0) for green)
        thickness: Line thickness for rectangles
        
    Returns:
        numpy.ndarray: Frame with detections drawn
    """
    if detections is None:
        return frame
    
    frame_copy = frame.copy()
    frame_height, frame_width = frame_copy.shape[:2]
    
    for detection in detections.detections:
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(
            detection.bounding_box, frame_width, frame_height
        )
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
    
    return frame_copy


def draw_segmasks_on_frame(frame, segmentations, color, alpha=0.8):
    """
    Draw segmentation masks on a frame with transparency.
    
    Args:
        frame: OpenCV image (numpy array)
        segmentations: fo.Detections object
        color: BGR color tuple (e.g., (0, 255, 0) for green)
        alpha: Transparency factor (0.0 to 1.0), where 1.0 is fully opaque
        
    Returns:
        numpy.ndarray: Frame with segmentation masks drawn
    """
    if segmentations is None:
        return frame
    
    frame_copy = frame.copy()
    frame_height, frame_width = frame_copy.shape[:2]
    
    # Create a colored overlay for all masks
    overlay = frame_copy.copy()
    
    for segmentation in segmentations.detections:
        # Get the mask
        mask = segmentation.mask

        # this is relative to the bbox
        bbox = segmentation.bounding_box
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(bbox, frame_width, frame_height)
        mask = fit_mask_to_bbox(mask, (y2-y1, x2-x1))

        # make it relative to the frame
        mask_framed = np.zeros((frame_height, frame_width), bool)
        mask_framed[y1:y2, x1:x2] = mask

        # Resize mask if dimensions don't match
        mask_height, mask_width = mask_framed.shape[:2]
        if (mask_height != frame_height) or (mask_width != frame_width):
            mask_framed = cv2.resize(mask_framed, (frame_width, frame_height), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Create binary mask (foreground pixels)
        binary_mask = (mask_framed > 0).astype(np.uint8)
        
        # Create colored mask
        colored_mask = np.zeros_like(frame_copy)
        colored_mask[:, :] = color
        
        # Apply mask to overlay
        mask_3d = np.stack([binary_mask] * 3, axis=2)
        overlay = np.where(mask_3d, colored_mask, overlay)
    
    # Blend overlay with original frame
    result = cv2.addWeighted(frame_copy, 1 - alpha, overlay, alpha, 0)
    
    return result


def visualize_bbox_detections(frame_prev, detections_prev, frame_curr, detections_curr, detections_prop):
    """
    Visualize ground truth and propagated detections on separate windows.
    
    Args:
        frame: OpenCV image (numpy array)
        detections_gt: Ground truth detections (fo.Detections)
        detections_prop: Propagated detections (fo.Detections)
    """
    # frame_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB)
    # frame_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2RGB)

    # Previous frame
    frame_prev = draw_bboxes_on_frame(frame_prev, detections_prev, (0, 255, 0))
    # frame_prev = cv2.resize(frame_prev, (0, 0), fx=0.25, fy=0.25)  # downsample to 1/4 size
    cv2.imshow("Previous Frame (green)", frame_prev)

    # Current frame
    frame_curr = draw_bboxes_on_frame(frame_curr, detections_curr, (0, 255, 0))
    # Propagated detections
    frame_prop = draw_bboxes_on_frame(frame_curr, detections_prop, (0, 0, 255))
    # frame_prop = cv2.resize(frame_prop, (0, 0), fx=0.25, fy=0.25)  # downsample to 1/4 size
    cv2.imshow("Propagated Detections (red) on new frame", frame_prop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_segmentation_detections(frame_prev, segmentations_prev, frame_curr, segmentations_curr, segmentations_prop, wait_timeout=0):
    """
    Visualize ground truth and propagated segmentations on separate windows.
    
    Args:
        frame_prev: Previous frame (OpenCV image, numpy array)
        segmentations_prev: Ground truth segmentations from previous frame (fo.Detections)
        frame_curr: Current frame (OpenCV image, numpy array)
        segmentations_curr: Ground truth segmentations from current frame (fo.Detections)
        segmentations_prop: Propagated segmentations (fo.Detections)
        wait_timeout: Timeout in milliseconds (0 = wait indefinitely, >0 = timeout)
    """
    # Previous frame with green masks
    frame_prev_viz = draw_segmasks_on_frame(frame_prev, segmentations_prev, (0, 255, 0), alpha=0.8)
    cv2.imshow("Previous Frame (green=GT)", frame_prev_viz)

    # Current frame with ground truth (green) and propagated (red) masks
    frame_curr_gt = draw_segmasks_on_frame(frame_curr, segmentations_curr, (0, 255, 0), alpha=0.8)
    cv2.imshow("Current Frame: (green=GT)", frame_curr_gt)
    frame_curr_prop = draw_segmasks_on_frame(frame_curr, segmentations_prop, (0, 0, 255), alpha=0.8)
    cv2.imshow("Current Frame: (red=Propagated)", frame_curr_prop)

    # Wait for key press with optional timeout
    key = cv2.waitKey(wait_timeout)
    # Clean up windows
    cv2.destroyAllWindows()
    # Give the system a moment to process window destruction
    cv2.waitKey(1)


if __name__ == "__main__":
    # Load the dataset
    dataset = fo.load_dataset("basketball_frames")

    PROP_IDS = [
        ("6939cb1e464fecaa395fb631", "6939cb1e464fecaa395fb633"),
        ("6939cb1f464fecaa395fb682", "6939cb1f464fecaa395fb683"),
        ("6939cb18464fecaa395fb3e5", "6939cb17464fecaa395fb3da")
    ]
    ANNOTATION_FIELD = "sam"

    for prop_from_id, prop_to_id in PROP_IDS:
        prop_from_sample = dataset[prop_from_id]
        prop_to_sample = dataset[prop_to_id]

        prev_frame = cv2.imread(prop_from_sample.filepath)
        curr_frame = cv2.imread(prop_to_sample.filepath)

        # prop_detections = propagate_bboxes_with_grabcut(
        # prop_detections = propagate_bboxes_with_densecrf(
        # prop_segmentations = propagate_segmentations_with_grabcut(
        prop_segmentations = propagate_segmentations_with_densecrf(
            prev_frame,
            curr_frame,
            prop_from_sample[ANNOTATION_FIELD]
        )
        visualize_segmentation_detections(
            prev_frame, prop_from_sample[ANNOTATION_FIELD],
            curr_frame, prop_to_sample[ANNOTATION_FIELD],
            prop_segmentations,
            wait_timeout=30000)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
