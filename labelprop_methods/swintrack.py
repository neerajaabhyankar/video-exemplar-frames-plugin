import os
import sys
import logging
from typing import Tuple, Union, Optional, Any
import numpy as np
import cv2

import fiftyone as fo

from utils import normalized_bbox_to_pixel_coords

logger = logging.getLogger(__name__)


class PropagatorSwinTrack:
    def __init__(self):
        self.swintrack_weights_path = None
        self.swintrack_config = None
        self.source_trackers_features: list[Tuple[Any, np.ndarray, str]] = []  # a tracker for each source detection
        self.template_size = None
        self.search_size = None
        self.normalize = None
        self.to_tensor = None
        self.device = None

    def parse_config(self):
        import torchvision.transforms as transforms
        self.template_size = tuple(self.swintrack_config['data']['template_size'])  # [112, 112]
        self.search_size = tuple(self.swintrack_config['data']['search_size'])  # [224, 224]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        
    def setup(self):
        """
        Setup SwinTrack model.
        
        Installation:
        1. git clone https://github.com/LitingLin/SwinTrack.git
        2. download SwinTrack-T-M.pth
           from https://drive.google.com/drive/folders/161lYU1-LVcVNwk-83eMhb-BsbOkK-pJa
           to SwinTrack/weights/
        """
        import torch
        import torchvision.transforms as transforms

        if (self.swintrack_weights_path is None) or (self.swintrack_config is None):
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
            self.swintrack_weights_path = weights_path
        
            # Load config using SwinTrack's YAML loader (handles !include tags)
            config_path = os.path.join(swintrack_path, "config", "SwinTrack", "Tiny", "config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"SwinTrack config not found at {config_path}"
                )
            
            from miscellanies.yaml_ops import load_yaml
            self.swintrack_config = load_yaml(config_path)
            self.parse_config()

            logger.info(f"Successfully found SwinTrack config and weights")
        
        # Build model
        from models.methods.SwinTrack.builder import build_swin_track_main_components
        from models.backbone.builder import build_backbone
        from models.head.builder import build_head
        from core.run.event_dispatcher.register import EventRegister
        
        # Build components
        backbone = build_backbone(self.swintrack_config, load_pretrained=False)
        encoder, decoder, out_norm, z_backbone_out_stage, x_backbone_out_stage, z_input_projection, x_input_projection, z_pos_enc, x_pos_enc = \
            build_swin_track_main_components(
                self.swintrack_config, num_epochs=1, iterations_per_epoch=1, 
                event_register=EventRegister('inference/'), has_training_run=False
            )
        head = build_head(self.swintrack_config)
    
        from models.methods.SwinTrack.network import SwinTrack
        logger.info(f"Creating new SwinTrack instance")
        model = SwinTrack(
            backbone, encoder, decoder, out_norm, head,
            z_backbone_out_stage, x_backbone_out_stage,
            z_input_projection, x_input_projection,
            z_pos_enc, x_pos_enc
        )
    
        # Load weights
        checkpoint = torch.load(self.swintrack_weights_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Filter out unexpected keys (like trajectory tokens that may not be in the inference model)
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict:
                if model_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    logger.warning(f"Skipping {k} due to shape mismatch: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
            else:
                logger.warning(f"Skipping unexpected key: {k}")
            
        # Load with strict=False to allow missing keys
        model.load_state_dict(filtered_state_dict, strict=False)
            
        # Determine device
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = device
        model.to(self.device)
        model.eval()
        
        return model

    
    def register_source_frame(self, source_frame, source_detections):
        """
        Register the source frame and detections with SwinTrack.
        
        Args:
            source_frame: The source frame (numpy array, BGR format)
            source_detections: The detections from source_frame (fo.Detections)
        """
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        
        source_height, source_width = source_frame.shape[:2]
        source_frame_rgb = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
        
        self.source_trackers_features = []
        
        if not hasattr(source_detections, "detections"):
            return
        
        for detection in source_detections.detections:
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
            template_pil = Image.fromarray(template_crop).resize(self.template_size, Image.BILINEAR)
            template_tensor = self.normalize(self.to_tensor(template_pil)).unsqueeze(0)
            
            try:
                # Create a new tracker instance for each detection
                # TODO(neeraja): separate the setup and init
                model = self.setup()
                with torch.no_grad():
                    z_feat = model.initialize(template_tensor.to(self.device))
                
                self.source_trackers_features.append(
                    (model, z_feat, detection.label if hasattr(detection, 'label') else None)
                )
            except Exception as e:
                logger.warning(f"SwinTrack initialization failed for detection: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def propagate_to_target_frame(self, target_frame):
        """
        Propagate the source detections to the target frame.
        
        Args:
            target_frame: The target frame (numpy array, BGR format)
        
        Returns:
            fo.Detections: New detections for the target frame
        """
        import torch
        from PIL import Image
        
        propagated_detections = []
        target_frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
        
        # Prepare search region from target frame (use full target frame as search region)
        search_pil = Image.fromarray(target_frame_rgb).resize(self.search_size, Image.BILINEAR)
        search_tensor = self.normalize(self.to_tensor(search_pil)).unsqueeze(0)
        
        for model, z_feat, source_label in self.source_trackers_features:
            with torch.no_grad():
                # Track on search frame
                output = model.track(z_feat, search_tensor.to(self.device))
                
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
                    # Fallback: skip this detection
                    logger.warning("SwinTrack output missing 'bbox' key, skipping detection")
                    continue
            else:
                # Fallback: skip this detection
                logger.warning(f"SwinTrack output is not a dict, got {type(output)}, skipping detection")
                continue
            
            # Create new detection
            new_detection = fo.Detection(
                bounding_box=new_bbox,
                label=source_label,
            )
            propagated_detections.append(new_detection)
        
        return fo.Detections(detections=propagated_detections)
