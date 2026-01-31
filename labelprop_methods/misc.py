import os
import sys
import logging
from typing import Tuple, Union, Optional, Any
import numpy as np
import cv2

import fiftyone as fo

from utils import fit_mask_to_bbox, normalized_bbox_to_pixel_coords

logger = logging.getLogger(__name__)


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

    if not hasattr(source_detections, "detections"):
        return fo.Detections(detections=propagated_detections)
    
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
            logger.warning("Warning: No contour found for segmentation")
            new_bbox = (0, 0, 0, 0)

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


def propagate_detections_with_densecrf(target_frame, source_detections):
    """
    Propagate detections from source_detections to target_frame using PyDenseCRF.
    Args:
        target_frame: The target frame to propagate to
        source_detections: The detections from source_frame (fo.Detections)

    Returns:
        fo.Detections: New detections for the target frame
    """
    import pydensecrf.densecrf as dcrf

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
    n_points = target_width * target_height
    
    propagated_detections = []
    if not hasattr(source_detections, "detections"):
        return fo.Detections(detections=propagated_detections)
    
    for detection in source_detections.detections:
        # Get source bbox and convert to pixel coordinates
        source_bbox = detection.bounding_box
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(source_bbox, target_width, target_height)
        
        # Get the mask from the source detection (optional)
        source_mask = detection.mask
        
        # Create unary potential
        unary = np.zeros((2, n_points), dtype=np.float32)  # 2 classes: bg, fg
        
        if source_mask is not None:
            # Case 1: Has mask - use mask-based initialization
            # Fit mask to bbox size
            source_mask_fitted = fit_mask_to_bbox(source_mask, (y2 - y1, x2 - x1))
            
            # Place mask in full frame coordinates
            source_mask_framed = np.zeros((target_height, target_width), dtype=np.float32)
            source_mask_framed[y1:y2, x1:x2] = source_mask_fitted.astype(np.float32)
            
            # Normalize mask to 0-1 range if needed
            if source_mask_framed.max() > 1.0:
                source_mask_framed = source_mask_framed / 255.0
            
            # Convert mask to probabilities
            mask_flat = source_mask_framed.flatten()
            
            # Set probabilities based on mask values
            unary[0, :] = 1.0 - mask_flat  # background probability
            unary[1, :] = mask_flat  # foreground probability
            
            # Apply temperature to soften/harden unary confidence
            if UNARY_TEMPERATURE != 1.0:
                unary = np.power(unary, 1.0 / UNARY_TEMPERATURE)
            
            # Add small epsilon to avoid log(0)
            unary = np.clip(unary, EPSILON, 1.0)
            
            # Normalize
            unary_sum = unary.sum(axis=0, keepdims=True)
            unary = unary / (unary_sum + EPSILON)
            
            # Convert to negative log probabilities
            unary = -np.log(unary)
        else:
            # Case 2: No mask - use bbox-based initialization
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
            
            # Convert to negative log probabilities
            unary = -np.log(unary + EPSILON)
        
        # Create DenseCRF
        d = dcrf.DenseCRF2D(target_width, target_height, 2)
        
        # Set unary potential
        d.setUnaryEnergy(unary)
        
        # Add pairwise Gaussian potential (spatial smoothness)
        d.addPairwiseGaussian(sxy=DENSECRF_SPATIAL_SMOOTHNESS, compat=DENSECRF_COMPAT)
        
        # Add pairwise bilateral potential (color-dependent)
        # Convert BGR to RGB for pydensecrf
        rgb_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
        d.addPairwiseBilateral(sxy=DENSECRF_BILATERAL_SPATIAL_SMOOTHNESS, 
                              srgb=DENSECRF_BILATERAL_COLOR_SMOOTHNESS, 
                              rgbim=rgb_frame, 
                              compat=DENSECRF_BILATERAL_COMPAT)
        
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
            logger.warning("Warning: No contour found for detection")
            new_bbox = (0, 0, 0, 0)
        
        # Extract mask relative to new bbox (if mask was provided)
        refined_mask_fitted = None
        if source_mask is not None:
            x1_new, y1_new, x2_new, y2_new = normalized_bbox_to_pixel_coords(new_bbox, target_width, target_height)
            refined_mask_fitted = refined_mask[y1_new:y2_new, x1_new:x2_new]
        
        # Create new detection
        new_detection = fo.Detection(
            bounding_box=new_bbox,
            mask=refined_mask_fitted,
            label=detection.label if hasattr(detection, 'label') else None,
        )
        propagated_detections.append(new_detection)
    
    return fo.Detections(detections=propagated_detections)


def propagate_detections_cv2_ot(source_frame, target_frame, source_detections):
    """
    Propagate detections from source_detections to target_frame using Object Tracking Algorithm
    """
    source_height, source_width = source_frame.shape[:2]
    target_height, target_width = target_frame.shape[:2]
    
    propagated_detections = []
    if not hasattr(source_detections, "detections"):
        return fo.Detections(detections=propagated_detections)
    
    tracker = cv2.TrackerCSRT_create()
    # tracker = cv2.legacy.TrackerMedianFlow_create()
    for detection in source_detections.detections:
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(
            detection.bounding_box, source_width, source_height
        )
        tracker.init(source_frame, (x1, y1, x2 - x1, y2 - y1))
        try:
            ok, (x, y, w, h) = tracker.update(target_frame)
        except:
            logger.warning("Warning: Tracker failed")
            x, y, w, h = (0, 0, 0, 0)
        new_bbox = [x / target_width, y / target_height, w / target_width, h / target_height]
        new_detection = fo.Detection(
            bounding_box=new_bbox,
            label=detection.label,
        )
        propagated_detections.append(new_detection)
    return fo.Detections(detections=propagated_detections)



def setup_per_sam():
    """
    1. git clone https://github.com/ZrrSkywalker/Personalize-SAM
       (no new requirements)
    2. comment out attn_sim and target_embedding in predictor.py::258-259
    """
    import torch
    import numpy as np
    from torch.nn import functional as F
    
    # Add Personalize-SAM directory to path to import per_segment_anything
    personalize_sam_path = os.path.join(os.path.dirname(__file__), "../Personalize-SAM")
    if personalize_sam_path not in sys.path:
        sys.path.insert(0, personalize_sam_path)
    
    from per_segment_anything import SamPredictor
    logger.info(f"Successfully imported Personalize-SAM")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # --- Load SAM backbone ---
    try:
        # First, try to get the underlying model from FiftyOne wrapper
        zoo_model = foz.load_zoo_model("segment-anything-vitb-torch")
        sam_model = zoo_model._model
        sam_model.to(device)
        sam_model.eval()
        logger.info(f"SAM model loaded on device: {device}")
    except Exception as e:
        logger.warning(f"Warning: Could not extract model from FiftyOne zoo model: {e}")
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

    if not hasattr(source_detections, "detections"):
        return fo.Detections(detections=propagated_segmentations)
    
    for segmentation in source_detections.detections:
        # Get the mask from the source segmentation (relative to bbox)
        source_mask = segmentation.mask

        if source_mask is None:
            print("Warning: No mask found for segmentation")
            continue
        
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

