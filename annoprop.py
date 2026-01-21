from typing import Tuple, Union, Optional
import logging
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

import fiftyone as fo

logger = logging.getLogger(__name__)

from utils import fit_mask_to_bbox, normalized_bbox_to_pixel_coords, evaluate, evaluate_matched
from annoprop_algos import (
    propagate_detections_with_grabcut, 
    propagate_detections_with_densecrf, 
    propagate_detections_cv2_ot,
    propagate_segmentations_with_persam,
    propagate_detections_with_siamese,
    propagate_detections_with_swintrack,
)

def propagate_detections_no_op(
    target_frame: np.ndarray,
    source_detections: fo.Detections,
) -> fo.Detections:
    """
    Propagate detections as-is (Do nothing).
    
    Args:
        target_frame: Target frame image (unused, kept for API consistency)
        source_detections: Source detections to propagate
        
    Returns:
        The source detections unchanged
    """
    return source_detections


def propagate_annotations(
    view: Union[fo.core.dataset.Dataset, fo.core.view.DatasetView],
    exemplar_frame_field: str,
    input_annotation_field: str,
    output_annotation_field: str,
    evaluate_propagation: Optional[bool] = True,
) -> dict[str, float]:
    """
    Propagate annotations from exemplar frames to all the frames.
    Args:
        view: The view to propagate annotations from
        exemplar_frame_field: The field name in which the exemplar frame assignments are stored
        input_annotation_field: The field name of the annotation to copy from the exemplar frame field
        output_annotation_field: The field name of the annotation to save to the target frame
        evaluate_propagation: Whether to evaluate the propagation against
                              the input annotation field present in the propagation targets.
    """
    scores = {}

    for sample in view:
        if sample[exemplar_frame_field]["is_exemplar"]:
            sample[output_annotation_field] = sample[input_annotation_field]
        elif len(sample[exemplar_frame_field]["exemplar_assignment"]) > 0:
            exemplar_frame_ids = sample[exemplar_frame_field]["exemplar_assignment"]

            # TODO(neeraja): handle multiple exemplar frames for the same sample
            exemplar_sample = view[exemplar_frame_ids[0]]
            exemplar_frame = cv2.imread(exemplar_sample.filepath)
            exemplar_detections = exemplar_sample[input_annotation_field]

            sample_frame = cv2.imread(sample.filepath)
            # propagated_detections = propagate_detections_no_op(sample_frame, exemplar_detections)
            # propagated_detections = propagate_detections_with_grabcut(sample_frame, exemplar_detections)
            # propagated_detections = propagate_detections_with_densecrf(sample_frame, exemplar_detections)
            # propagated_detections = propagate_detections_cv2_ot(exemplar_frame, sample_frame, exemplar_detections)
            # propagated_detections = propagate_segmentations_with_persam(exemplar_frame, sample_frame, exemplar_detections)
            propagated_detections = propagate_detections_with_siamese(exemplar_frame, sample_frame, exemplar_detections)
            # propagated_detections = propagate_detections_with_swintrack(exemplar_frame, sample_frame, exemplar_detections)
            sample[output_annotation_field] = propagated_detections

            # If the sample already has an input annotation field, evaluate against it
            if evaluate_propagation and sample[input_annotation_field]:
                original_detections = sample[input_annotation_field]
                # TODO(neeraja): decouple the matching and the evaluation
                sample_score = evaluate(original_detections, propagated_detections)
                logger.debug(f"Sample {sample.id} score: {sample_score}")
                scores[sample.id] = sample_score
        sample.save()
    
    return scores
