import os
import tempfile
import shutil
from pathlib import Path
import logging
from typing import Tuple, Union, Optional, Any
import numpy as np
import cv2
from collections import OrderedDict

import fiftyone as fo

from utils import normalized_bbox_to_pixel_coords, fit_mask_to_bbox

logger = logging.getLogger(__name__)


class PropagatorSAM2:
    def __init__(self, model_cfg=None, checkpoint=None):
        """
        Initialize SAM2 propagator.
        1. Install SAM2 from https://github.com/facebookresearch/segment-anything-2
        2. Download the config and checkpoint to the installed location under weights/
        """
        # SAM2 uses Hydra config loading, which expects config names relative to its search path
        # Use the config name that exists in the SAM2 package, not an absolute path
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.checkpoint = "weights/sam2.1_hiera_tiny.pt"
        self.sam2_predictor = None
        self.inference_state = None
        self.setup()
        self.preds_dict = OrderedDict()
        self.label_type = "bounding_box"

    def setup(self):
        import torch
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        import sam2
        from sam2.build_sam import build_sam2_video_predictor

        package_dir = os.path.dirname(os.path.abspath(sam2.__file__))
        checkpoint_path = os.path.join(package_dir, self.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM2 checkpoint not found at {checkpoint_path}")
        
        self.sam2_predictor = build_sam2_video_predictor(self.model_cfg, checkpoint_path, device=device)
        logger.info("SAM2 predictor initialized successfully")
    
    def path_list_to_dir(self, image_path_list):
        """
        Convert a list of image paths to a temporary directory
        using simlinks, maintaining the order of the images.

        Args:
            image_path_list: List of image file paths
        Returns:
            Temporary directory path
        """
        tmpdir = Path(tempfile.mkdtemp())
        for ii, pp in enumerate(image_path_list):
            tmp_path = tmpdir / f"{ii:06d}{Path(pp).suffix}"
            tmp_path.symlink_to(Path(pp).resolve())
        logger.info(f"Created temporary directory {tmpdir} with {len(image_path_list)} frames")
        return tmpdir
    
    def initialize(self, frame_path_list):
        """
        Initialize the inference state with the frames list.
        Args:
            frame_path_list: List of frame file paths ordered by frame number
        """
        # TODO(neeraja): handle video files
        frames_dir = self.path_list_to_dir(frame_path_list)            
        self.inference_state = self.sam2_predictor.init_state(str(frames_dir))
        self.preds_dict.clear()
        for idx, frame_path in enumerate(frame_path_list):
            self.preds_dict[frame_path] = None
        shutil.rmtree(frames_dir)
        logger.info(f"Inference state initialized with {len(frame_path_list)} frames; cleaned up temporary directory {frames_dir}")

    def register_source_frame(self, source_filepath, source_detections):
        """
        Register the source frame and detections with SAM2.
        
        Args:
            source_filepath: The source frame file path
            source_detections: The detections from source_frame (fo.Detections)
        """
        if not hasattr(source_detections, "detections"):
            logger.warning(f"Source detections is either empty, or not a fo.Detections object: {source_detections}")
            return
        
        source_frame_idx = list(self.preds_dict.keys()).index(source_filepath)
        logger.debug(f"Registering source frame {source_filepath} at index {source_frame_idx}")
        
        source_frame = cv2.imread(source_filepath)
        source_height, source_width = source_frame.shape[:2]
        
        for detection in source_detections.detections:
            # Get source bbox and convert to pixel coordinates
            source_bbox = detection.bounding_box
            x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(source_bbox, source_width, source_height)

            source_mask = detection.mask
            if source_mask is not None:
                self.label_type = "segmentation_mask"
                source_mask_fitted = fit_mask_to_bbox(source_mask, (y2-y1, x2-x1))
                # make it relative to the target frame
                source_mask_framed = np.zeros((source_height, source_width), bool)
                source_mask_framed[y1:y2, x1:x2] = source_mask_fitted
                source_mask_framed = source_mask_framed.astype(np.uint8)

                self.sam2_predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=source_frame_idx,
                    obj_id=detection.label,
                    mask=source_mask_framed
                )
                logger.debug(f"Added new segmentation mask: {detection.label}")
            else:               
                self.sam2_predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=source_frame_idx,
                    obj_id=detection.label,
                    box=[x1, y1, x2, y2]
                )
                logger.debug(f"Added new bounding box: {detection.label}")
            
    def propagate_to_all_frames(self):
        """
        Propagate detections to all frames in the inference state.
        Uses SAM2's propagate_in_video API.
        Returns:
            None
            Populates the preds_dict with predictions for each frame
        """
        propagated_to_count = 0
        for frame_idx, obj_ids, mask_logits in self.sam2_predictor.propagate_in_video(self.inference_state):
            target_filepath = list(self.preds_dict.keys())[frame_idx]
            target_frame = cv2.imread(target_filepath)
            target_height, target_width = target_frame.shape[:2]

            propagated_detections = []
            obj_ids = list(obj_ids)
            logger.debug(f"Found {len(obj_ids)} detections for frame {target_filepath}")
            for oi, obj_id in enumerate(obj_ids):
                logits = mask_logits[oi].squeeze(0)
                pred = (logits > 0).cpu().numpy().astype(np.uint8) * 255  # threshold at 0

                # Find new bbox from pred
                contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    new_bbox = [x / target_width, y / target_height, w / target_width, h / target_height]
                else:
                    logger.warning("Warning: No contour found for detection")
                    new_bbox = (0, 0, 0, 0)
                

                if self.label_type == "segmentation_mask":
                    x1_new, y1_new, x2_new, y2_new = normalized_bbox_to_pixel_coords(
                        new_bbox, target_width, target_height
                    )
                    mask_fitted = pred[y1_new:y2_new, x1_new:x2_new]
                    mask_fitted = (mask_fitted.astype(np.float32) / np.max(mask_fitted)).astype(np.uint8)
                else:
                    mask_fitted = None

                new_detection = fo.Detection(
                    bounding_box=new_bbox,
                    mask=mask_fitted,
                    label=obj_id,
                )
                propagated_detections.append(new_detection)
            
            logger.debug(f"Propagated {len(propagated_detections)} detections for frame {target_filepath}")
            self.preds_dict[target_filepath] = fo.Detections(detections=propagated_detections)
            propagated_to_count += 1
        
        logger.info(f"Propagated detections to {propagated_to_count} frames out of {len(self.preds_dict)}")
    
    def propagate_to_target_frame(self, target_filepath):
        """
        Propagate masks to target frame using SAM2.
        
        Args:
            target_filepath: Target frame file path
            
        Returns:
            fo.Detections: Propagated detections with masks
        """
        if self.inference_state is None:
            raise RuntimeError("Must call register_source_frame() before propagate_to_target_frame()")
        
        if os.path.abspath(target_filepath) not in self.preds_dict:
            logger.warning(f"Target frame {target_filepath} not found in predictions")
            return fo.Detections(detections=[])
        
        return self.preds_dict[target_filepath]