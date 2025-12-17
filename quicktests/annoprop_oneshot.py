import fiftyone as fo
import cv2
import numpy as np
from PIL import Image
from typing import Tuple

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

from eta.core.video import VideoProcessor
import fiftyone.zoo as foz


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


def setup_per_sam():
    """
    1. git clone https://github.com/ZrrSkywalker/Personalize-SAM
       (no new requirements)
    2. comment out attn_sim and target_embedding in predictor.py::258-259
    """
    import torch
    import numpy as np
    import os
    import sys
    from torch.nn import functional as F
    
    # Add Personalize-SAM directory to path to import per_segment_anything
    personalize_sam_path = os.path.join(os.path.dirname(__file__), "../Personalize-SAM")
    if personalize_sam_path not in sys.path:
        sys.path.insert(0, personalize_sam_path)
    
    from per_segment_anything import SamPredictor

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # --- Load SAM backbone ---
    try:
        # First, try to get the underlying model from FiftyOne wrapper
        zoo_model = foz.load_zoo_model("segment-anything-vitb-torch")
        sam_model = zoo_model._model
        sam_model.to(device)
        sam_model.eval()
        print(f"SAM model loaded on device: {device}")
    except Exception as e:
        print(f"Warning: Could not extract model from FiftyOne zoo model: {e}")
        breakpoint()

    # --- Create SamPredictor (PerSAM uses this directly) ---
    predictor = SamPredictor(sam_model)
    return predictor


def execute_per_sam(per_sam_predictor, source_frame, target_frame, source_mask):
    """
    Execute PerSAM on the source frame and target frame.
    Args:
        per_sam_predictor: PerSAM predictor
        source_frame: Source frame
        target_frame: Target frame same size as source frame
        source_mask: Source mask fit to the source frame
    Returns:
        target_mask: Target mask same size as target frame
    """
    import torch
    from torch.nn import functional as F

    # Convert source_frame to RGB if needed (OpenCV uses BGR)
    if len(source_frame.shape) == 3 and source_frame.shape[2] == 3:
        source_frame_rgb = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
    else:
        source_frame_rgb = source_frame
    
    # Convert source_mask to the format expected by predictor.set_image
    # predictor.set_image expects mask as (H, W, 3) RGB image or (1, 1, H, W) tensor
    if len(source_mask.shape) == 2:
        source_mask_rgb = np.stack([source_mask] * 3, axis=2)
    else:
        source_mask_rgb = source_mask
    
    # --- Step 1: Extract concept embedding (PerSAM logic) ---
    with torch.no_grad():
        # Set image and mask to extract features
        ref_mask = per_sam_predictor.set_image(source_frame_rgb, source_mask_rgb)
        ref_feat = per_sam_predictor.features.squeeze().permute(1, 2, 0)
        
        # Process mask to match feature dimensions
        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0:2], mode="bilinear")
        ref_mask = ref_mask.squeeze()[0]
        
        # Extract target feature embedding
        target_feat = ref_feat[ref_mask > 0]
        target_embedding = target_feat.mean(0).unsqueeze(0)
        target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        target_embedding = target_embedding.unsqueeze(0)
    
    # --- Step 2: Apply to target image ---
    # Convert target_frame to RGB if needed
    if len(target_frame.shape) == 3 and target_frame.shape[2] == 3:
        target_frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
    else:
        target_frame_rgb = target_frame
    
    with torch.no_grad():
        # Encode target image
        per_sam_predictor.set_image(target_frame_rgb)
        test_feat = per_sam_predictor.features.squeeze()
        
        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = target_feat @ test_feat
        
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = per_sam_predictor.model.postprocess_masks(
            sim,
            input_size=per_sam_predictor.input_size,
            original_size=per_sam_predictor.original_size
        ).squeeze()
        
        # Get location prior (simplified - you may want to implement point_selection)
        # For now, use threshold-based approach
        sim_normalized = (sim - sim.mean()) / torch.std(sim)
        sim_normalized = F.interpolate(
            sim_normalized.unsqueeze(0).unsqueeze(0), 
            size=(64, 64), 
            mode="bilinear"
        )
        attn_sim = sim_normalized.sigmoid_().unsqueeze(0).flatten(3)
        
        # Get top point for guidance
        topk_xy = torch.argmax(sim.flatten()).item()
        h_sim, w_sim = sim.shape
        topk_y = topk_xy // w_sim
        topk_x = topk_xy % w_sim
        topk_xy = np.array([[topk_x, topk_y]])
        topk_label = np.array([1])
        
        # Predict mask
        masks, scores, logits, _ = per_sam_predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=False,
            attn_sim=attn_sim,
            target_embedding=target_embedding
        )
        
        target_mask = masks[0].astype(np.uint8)
    
    return target_mask


def propagate_segmentations_with_persam(source_frame, target_frame, source_detections):
    """
    Propagate segmentations from source_frame to target_frame using DenseCRF.
    
    Args:
        source_frame: The source frame with segmentations (used for reference, not directly used)
        target_frame: The target frame to propagate to
        source_detections: The segmentations from source_frame (fo.Detections)
        
    Returns:
        fo.Detections: New detections with propagated masks for the target frame
    """

    target_height, target_width = target_frame.shape[:2]
    source_height, source_width = source_frame.shape[:2]
    n_points = target_width * target_height
    
    propagated_segmentations = []
    
    for segmentation in source_detections.detections:
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
        
        # TODO
        # use source_mask_framed
        # get refined_mask
        # refined_mask = source_mask_framed.astype(np.uint8)

        per_sam_predictor = setup_per_sam()
        refined_mask = execute_per_sam(per_sam_predictor, source_frame, target_frame, source_mask_framed)

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
        ("6939cb18464fecaa395fb3fc", "6939cb18464fecaa395fb3fb"),
        ("6939cb19464fecaa395fb487", "6939cb19464fecaa395fb488"),
        ("6939cb1e464fecaa395fb60a", "6939cb1e464fecaa395fb60b")
    ]
    ANNOTATION_FIELD = "sam"

    for prop_from_id, prop_to_id in PROP_IDS:
        prop_from_sample = dataset[prop_from_id]
        prop_to_sample = dataset[prop_to_id]

        prev_frame = cv2.imread(prop_from_sample.filepath)
        curr_frame = cv2.imread(prop_to_sample.filepath)

        prop_segmentations = propagate_segmentations_with_persam(
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
