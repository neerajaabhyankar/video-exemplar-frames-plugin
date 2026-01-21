from typing import Tuple, Union, Optional
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

import fiftyone as fo

from utils import fit_mask_to_bbox, normalized_bbox_to_pixel_coords, evaluate, evaluate_matched


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
            print("Warning: No contour found for detection")
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
            print("Warning: Tracker failed")
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
    import os
    import sys
    from torch.nn import functional as F
    
    # Add Personalize-SAM directory to path to import per_segment_anything
    personalize_sam_path = os.path.join(os.path.dirname(__file__), "../Personalize-SAM")
    if personalize_sam_path not in sys.path:
        sys.path.insert(0, personalize_sam_path)
    
    from per_segment_anything import SamPredictor
    print(f"Successfully imported Personalize-SAM")

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



# Cache for SiamFC tracker class and net_path to avoid reloading
_siamfc_tracker_class = None
_siamfc_net_path = None

def setup_siamfc():
    """
    Setup SiamFC tracker. 
    
    Installation options:
    1. (Recommended) Install via pip:
       pip install git+https://github.com/neerajaabhyankar/siamfc-pytorch.git
       
       Then download weights to the installed package location:
       - Find the installed location:
         python -c 'import siamfc; import os; print(os.path.dirname(os.path.abspath(siamfc.__file__)))'
       - Download siamfc_alexnet_e50.pth from:
         https://drive.google.com/file/d/1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4/view?usp=sharing
       - Place it in <installed_location>/weights/
       
    2. Or use local installation (for development):
       - Clone the repository locally
       - Download siamfc_alexnet_e50.pth from:
         https://drive.google.com/file/d/1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4/view?usp=sharing
       - Place it in siamfc-pytorch/weights/
    """
    import os
    import sys
    
    global _siamfc_tracker_class, _siamfc_net_path
    
    # If already loaded, create new tracker instance
    if _siamfc_tracker_class is not None and _siamfc_net_path is not None:
        return _siamfc_tracker_class(net_path=_siamfc_net_path)
    
    # Try importing from installed package first
    try:
        from siamfc import TrackerSiamFC
        print("Successfully imported SiamFC from installed package")
        installed_package = True
    except ImportError:
        # Fallback to local path for dev
        current_dir = os.path.dirname(os.path.abspath(__file__))
        siamfc_path = os.path.join(current_dir, "siamfc-pytorch")
        if not os.path.exists(siamfc_path):
            raise ImportError(
                "siamfc-pytorch not found. Please install it:\n"
                "pip install git+https://github.com/neerajaabhyankar/siamfc-pytorch.git\n"
                f"Or ensure local repository exists at {siamfc_path}"
            )
        sys.path.insert(0, siamfc_path)
        from siamfc import TrackerSiamFC
        print("Successfully imported SiamFC from local path")
        installed_package = False
    
    # Get weights path from the installed package location first
    weights_path = None
    
    if installed_package:
        # Try to find weights in installed package location
        import siamfc
        package_dir = os.path.dirname(os.path.abspath(siamfc.__file__))
        weights_path = os.path.join(package_dir, "weights", "siamfc_alexnet_e50.pth")
        weights_path = os.path.abspath(weights_path)
    
    # Fall back to local path for dev
    if not weights_path or not os.path.exists(weights_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "siamfc-pytorch", "weights", "siamfc_alexnet_e50.pth")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"SiamFC weights not found at {weights_path}. "
            "Please download siamfc_alexnet_e50.pth from:\n"
            "https://drive.google.com/file/d/1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4/view?usp=sharing\n"
            "and place it in the weights/ directory."
        )
    
    # Cache the class and net_path for future use
    _siamfc_tracker_class = TrackerSiamFC
    _siamfc_net_path = weights_path
    
    # Initialize tracker with net_path
    tracker = TrackerSiamFC(net_path=weights_path)
    print(f"Successfully initialized SiamFC tracker")
    
    return tracker


def propagate_detections_with_siamese(source_frame, target_frame, source_detections):
    """
    Propagate detections from source_frame to target_frame using Siamese Network (SiamFC).
    
    Args:
        source_frame: The source frame (numpy array, BGR format)
        target_frame: The target frame (numpy array, BGR format)
        source_detections: The detections from source_frame (fo.Detections)
    
    Returns:
        fo.Detections: New detections for the target frame
    """
    source_height, source_width = source_frame.shape[:2]
    target_height, target_width = target_frame.shape[:2]
    
    propagated_detections = []
    if not hasattr(source_detections, "detections"):
        return fo.Detections(detections=propagated_detections)
    
    # Convert frames to RGB (SiamFC expects RGB)
    source_frame_rgb = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
    target_frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
    
    for detection in source_detections.detections:
        # Get bounding box from source detection and convert to pixel coordinates
        source_bbox = detection.bounding_box
        x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(
            source_bbox, source_width, source_height
        )
        
        # Convert back to (x, y, width, height) format for SiamFC
        # SiamFC expects 1-indexed coordinates (x, y are 1-indexed)
        init_bbox = (
            x1 + 1, y1 + 1, # Convert to 1-indexed
            x2 - x1, y2 - y1,
        )
        
        try:
            # Create a new tracker instance for each detection
            tracker = setup_siamfc()
            
            # Initialize tracker with source frame and bbox
            tracker.init(source_frame_rgb, init_bbox)
            
            # Update tracker with target frame
            new_bbox = tracker.update(target_frame_rgb)
            
            # new_bbox is in (x, y, width, height) format, 1-indexed
            new_x, new_y, new_w, new_h = new_bbox
            
            # Convert from 1-indexed to 0-indexed, then to normalized coordinates
            new_bbox_normalized = [
                (new_x - 1) / target_width,  # Convert to 0-indexed and normalize
                (new_y - 1) / target_height,  # Convert to 0-indexed and normalize
                new_w / target_width,
                new_h / target_height
            ]
            
            # Ensure bbox is within image bounds
            new_bbox_normalized[0] = max(0.0, min(new_bbox_normalized[0], 1.0))
            new_bbox_normalized[1] = max(0.0, min(new_bbox_normalized[1], 1.0))
            new_bbox_normalized[2] = max(0.0, min(new_bbox_normalized[2], 1.0 - new_bbox_normalized[0]))
            new_bbox_normalized[3] = max(0.0, min(new_bbox_normalized[3], 1.0 - new_bbox_normalized[1]))
            
            # Create new detection
            new_detection = fo.Detection(
                bounding_box=new_bbox_normalized,
                label=detection.label if hasattr(detection, 'label') else None,
            )
            propagated_detections.append(new_detection)
            
        except Exception as e:
            print(f"Warning: SiamFC tracking failed for detection: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: use original bbox (or skip)
            continue
    
    return fo.Detections(detections=propagated_detections)



# Cache for SwinTrack model to avoid reloading
_swintrack_model = None
_swintrack_config = None
_swintrack_device = None

def setup_swintrack():
    """
    1. git clone https://github.com/LitingLin/SwinTrack.git
    2. download SwinTrack-T-M.pth
       from https://drive.google.com/drive/folders/161lYU1-LVcVNwk-83eMhb-BsbOkK-pJa
       to SwinTrack/weights/
    """
    import os
    import sys
    import torch
    import torchvision.transforms as transforms
    
    global _swintrack_model, _swintrack_config, _swintrack_device
    
    # If already loaded, return the model
    if _swintrack_model is not None:
        return _swintrack_model, _swintrack_config, _swintrack_device
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    swintrack_path = os.path.join(current_dir, "SwinTrack")
    if not os.path.exists(swintrack_path):
        raise FileNotFoundError(
            f"SwinTrack repository not found at {swintrack_path}. Please clone it:\n"
            "git clone https://github.com/LitingLin/SwinTrack.git"
        )
    sys.path.insert(0, swintrack_path)
    
    # Get weights path
    weights_path = os.path.join(current_dir, "SwinTrack", "weights", "SwinTrack-T-M.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"SwinTrack weights not found at {weights_path}. "
            "Please download SwinTrack-T-M.pth to SwinTrack/weights/"
        )
    
    # Load config using SwinTrack's YAML loader (handles !include tags)
    config_path = os.path.join(swintrack_path, "config", "SwinTrack", "Tiny", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"SwinTrack config not found at {config_path}"
        )
    
    from miscellanies.yaml_ops import load_yaml
    config = load_yaml(config_path)
    
    # Build model
    from models.methods.SwinTrack.builder import build_swin_track_main_components
    from models.backbone.builder import build_backbone
    from models.head.builder import build_head
    from core.run.event_dispatcher.register import EventRegister
    
    # Build components
    backbone = build_backbone(config, load_pretrained=False)
    encoder, decoder, out_norm, z_backbone_out_stage, x_backbone_out_stage, z_input_projection, x_input_projection, z_pos_enc, x_pos_enc = \
        build_swin_track_main_components(config, num_epochs=1, iterations_per_epoch=1, 
                                       event_register=EventRegister('inference/'), has_training_run=False)
    head = build_head(config)
    
    from models.methods.SwinTrack.network import SwinTrack
    model = SwinTrack(backbone, encoder, decoder, out_norm, head, 
                     z_backbone_out_stage, x_backbone_out_stage,
                     z_input_projection, x_input_projection,
                     z_pos_enc, x_pos_enc)
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Filter out unexpected keys (like trajectory tokens that may not be in the inference model)
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if model_state_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"Warning: Skipping {k} due to shape mismatch: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
        else:
            print(f"Warning: Skipping unexpected key: {k}")
    
    # Load with strict=False to allow missing keys
    model.load_state_dict(filtered_state_dict, strict=False)
    
    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    
    print(f"Successfully initialized SwinTrack on device: {device}")
    
    # Cache for future use
    _swintrack_model = model
    _swintrack_config = config
    _swintrack_device = device
    
    return model, config, device


def propagate_detections_with_swintrack(source_frame, target_frame, source_detections):
    """
    Propagate detections from source_frame to target_frame using SwinTrack.
    
    Args:
        source_frame: The source frame (numpy array, BGR format)
        target_frame: The target frame (numpy array, BGR format)
        source_detections: The detections from source_frame (fo.Detections)
    
    Returns:
        fo.Detections: New detections for the target frame
    """
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    
    source_height, source_width = source_frame.shape[:2]
    target_height, target_width = target_frame.shape[:2]
    
    propagated_detections = []
    if not hasattr(source_detections, "detections"):
        return fo.Detections(detections=propagated_detections)
    
    # Setup SwinTrack
    model, config, device = setup_swintrack()
    
    # Get template and search sizes from config
    template_size = tuple(config['data']['template_size'])  # [112, 112]
    search_size = tuple(config['data']['search_size'])  # [224, 224]
    
    # Image normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    # Convert frames to RGB
    source_frame_rgb = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
    target_frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
    
    for detection in source_detections.detections:
        try:
            # Get bounding box from source detection
            source_bbox = detection.bounding_box
            x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(
                source_bbox, source_width, source_height
            )
            
            # Crop template region from source frame (with padding)
            # Calculate center and size
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            
            # Add padding (area_factor similar to SiamFC)
            area_factor = 2.0
            w_padded = w + (area_factor - 1) * ((w + h) * 0.5)
            h_padded = h + (area_factor - 1) * ((w + h) * 0.5)
            
            # Crop template
            x1_crop = max(0, int(cx - w_padded / 2))
            y1_crop = max(0, int(cy - h_padded / 2))
            x2_crop = min(source_width, int(cx + w_padded / 2))
            y2_crop = min(source_height, int(cy + h_padded / 2))
            
            template_crop = source_frame_rgb[y1_crop:y2_crop, x1_crop:x2_crop]
            template_pil = Image.fromarray(template_crop).resize(template_size, Image.BILINEAR)
            template_tensor = normalize(to_tensor(template_pil)).unsqueeze(0).to(device)
            
            # Prepare search region from target frame (center crop or full image)
            # For simplicity, use full target frame as search region
            search_pil = Image.fromarray(target_frame_rgb).resize(search_size, Image.BILINEAR)
            search_tensor = normalize(to_tensor(search_pil)).unsqueeze(0).to(device)
            
            # Keep tensors in (B, C, H, W) format for backbone
            # The backbone expects standard image format when reshape=False
            
            with torch.no_grad():
                # Initialize with template
                z_feat = model.initialize(template_tensor)
                
                # Track on search frame
                output = model.track(z_feat, search_tensor)
            
            # Extract bounding box from output
            # Output format: dict with 'bbox' key (normalized coordinates)
            if isinstance(output, dict):
                if 'bbox' in output:
                    # bbox is in format (B, H, W, 4) or (B, H*W, 4)
                    bbox_pred = output['bbox']
                    if len(bbox_pred.shape) == 4:
                        # Take center location (H//2, W//2)
                        h_idx, w_idx = bbox_pred.shape[1] // 2, bbox_pred.shape[2] // 2
                        bbox_normalized = bbox_pred[0, h_idx, w_idx].cpu().numpy()
                    else:
                        # Take center of flattened
                        center_idx = bbox_pred.shape[1] // 2
                        bbox_normalized = bbox_pred[0, center_idx].cpu().numpy()
                    
                    # Convert from CXCYWH to XYXY if needed
                    # Config says format is "CXCYWH" with range [0, 1]
                    cx, cy, w_norm, h_norm = bbox_normalized
                    x1_norm = cx - w_norm / 2
                    y1_norm = cy - h_norm / 2
                    x2_norm = cx + w_norm / 2
                    y2_norm = cy + h_norm / 2
                    
                    # Clamp to [0, 1]
                    x1_norm = max(0.0, min(1.0, x1_norm))
                    y1_norm = max(0.0, min(1.0, y1_norm))
                    x2_norm = max(0.0, min(1.0, x2_norm))
                    y2_norm = max(0.0, min(1.0, y2_norm))
                    
                    new_bbox = [x1_norm, y1_norm, x2_norm - x1_norm, y2_norm - y1_norm]
                else:
                    # Fallback: use original bbox
                    new_bbox = source_bbox
            else:
                # Fallback: use original bbox
                new_bbox = source_bbox
            
            # Create new detection
            new_detection = fo.Detection(
                bounding_box=new_bbox,
                label=detection.label if hasattr(detection, 'label') else None,
            )
            propagated_detections.append(new_detection)
            
        except Exception as e:
            print(f"Warning: SwinTrack tracking failed for detection: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: use original bbox (or skip)
            continue
    
    return fo.Detections(detections=propagated_detections)