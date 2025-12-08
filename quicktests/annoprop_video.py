import fiftyone as fo
import cv2
import numpy as np
from PIL import Image


from eta.core.video import VideoProcessor

TEST_FREQUENCY = 20

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


def propagate_detections_with_grabcut(source_frame, target_frame, source_detections):
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


def normalized_bbox_to_pixel_coords(bbox, width, height):
    """
    Convert normalized bounding box [x, y, width, height] to pixel coordinates.
    
    Args:
        bbox: Normalized bounding box [x, y, width, height]
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        tuple: (x1, y1, x2, y2) pixel coordinates
    """
    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)
    x2 = int((bbox[0] + bbox[2]) * width)
    y2 = int((bbox[1] + bbox[3]) * height)
    return (x1, y1, x2, y2)


def draw_detections_on_frame(frame, detections, color, thickness=2):
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


def visualize_detections(frame_prev, detections_prev, frame_curr, detections_curr, detections_prop):
    """
    Visualize ground truth and propagated detections on separate windows.
    
    Args:
        frame: OpenCV image (numpy array)
        detections_gt: Ground truth detections (fo.Detections)
        detections_prop: Propagated detections (fo.Detections)
    """
    frame_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB)
    frame_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2RGB)

    # Previous frame
    frame_prev = draw_detections_on_frame(frame_prev, detections_prev, (0, 255, 0))
    frame_prev = cv2.resize(frame_prev, (0, 0), fx=0.25, fy=0.25)  # downsample to 1/4 size
    cv2.imshow("Previous Frame (green)", frame_prev)

    # Current frame
    frame_curr = draw_detections_on_frame(frame_curr, detections_curr, (0, 255, 0))
    # Propagated detections
    frame_prop = draw_detections_on_frame(frame_curr, detections_prop, (0, 0, 255))
    frame_prop = cv2.resize(frame_prop, (0, 0), fx=0.25, fy=0.25)  # downsample to 1/4 size
    cv2.imshow("Propagated Detections (red) on new frame", frame_prop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load the dataset
    dataset = fo.load_dataset("quickstart-video")

    SAMPLE_ID = "69370fe198f5892985959a8b"
    for sample, detections_object_list in zip(
        dataset, dataset.values("frames.detections")
    ):
        if sample.id == SAMPLE_ID:
            break

    prev_frame_found = False
    frame_prev = None
    detections_prev = None
    frame_curr = None
    detections_curr = None

    with VideoProcessor(
        sample.filepath
    ) as vp:
        for frame, detections_object in zip(vp, detections_object_list):
            if vp.frame_number % TEST_FREQUENCY == TEST_FREQUENCY-1:
                frame_prev = frame
                detections_prev = detections_object
                prev_frame_found = True
            elif prev_frame_found and vp.frame_number % TEST_FREQUENCY == 0:
                frame_curr = frame
                detections_curr = detections_object

                prop_detections = propagate_detections_with_grabcut(
                    frame_prev,
                    frame_curr,
                    detections_prev
                )
                visualize_detections(frame_prev, detections_prev, frame_curr, detections_curr, prop_detections)

            else:
                prev_frame_found = False
