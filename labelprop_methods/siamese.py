import os
import sys
import logging
from typing import Tuple, Union, Optional, Any
import numpy as np
import cv2

import fiftyone as fo

from suc_utils import normalized_bbox_to_pixel_coords

logger = logging.getLogger(__name__)


class PropagatorSiamFC:
    def __init__(self):
        self.siamfc_tracker_class = None
        self.siamfc_net_path = None
        self.source_trackers: list[Tuple[Any, str]] = []  # a tracker for each source detection

    def setup(self):
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

        Returns:
            TrackerSiamFC: The SiamFC tracker instance
        """
        if (self.siamfc_tracker_class is None) or (self.siamfc_net_path is None):
            
            # Try importing from installed package first
            try:
                from siamfc import TrackerSiamFC
                logger.info("Successfully imported SiamFC from installed package")
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
            self.siamfc_tracker_class = TrackerSiamFC
            self.siamfc_net_path = weights_path

            logger.info(f"Successfully found SiamFC weights")
        
        logger.info(f"Creating new SiamFC tracker instance")
        return self.siamfc_tracker_class(net_path=self.siamfc_net_path)
    
    def register_source_frame(self, source_frame, source_detections):
        """
        Register the source frame and detections with the SiamFC tracker.
        
        Args:
            source_frame: The source frame (numpy array, BGR format)
            source_detections: The detections from source_frame (fo.Detections)
        """
        source_height, source_width = source_frame.shape[:2]
        source_frame_rgb = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)

        self.source_trackers = []
        
        if not hasattr(source_detections, "detections"):
            return
    
        for detection in source_detections.detections:
            source_bbox = detection.bounding_box
            x1, y1, x2, y2 = normalized_bbox_to_pixel_coords(
                source_bbox, source_width, source_height
            )
            # Convert back to (x, y, width, height) format for SiamFC
            # SiamFC expects 1-indexed coordinates (x, y are 1-indexed)
            init_bbox = (
                x1 + 1, y1 + 1,
                x2 - x1, y2 - y1,
            )
        
            try:
                # Create a new tracker instance for each detection
                # TODO(neeraja): separate the setup and init
                tracker = self.setup()
                tracker.init(source_frame_rgb, init_bbox)
                self.source_trackers.append((tracker, detection.label if hasattr(detection, 'label') else None))
            except Exception as e:
                print(f"Warning: SiamFC tracking failed for detection: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def propagate_to_target_frame(self, target_frame):
        """
        Propagate the source detections to the target frame.
        
        Args:
            target_frame: The target frame (numpy array, BGR format)
        """
        propagated_detections = []
        target_height, target_width = target_frame.shape[:2]
        target_frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
    
        for di, (tracker, source_label) in enumerate(self.source_trackers):
            new_bbox = tracker.update(target_frame_rgb)            
            # new_bbox is in (x, y, width, height) format, 1-indexed
            new_x, new_y, new_w, new_h = new_bbox
            # Convert from 1-indexed to 0-indexed, then to normalized coordinates
            new_bbox_normalized = [
                (new_x - 1) / target_width,
                (new_y - 1) / target_height,
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
                label=source_label,
            )
            propagated_detections.append(new_detection)
        
        return fo.Detections(detections=propagated_detections)
