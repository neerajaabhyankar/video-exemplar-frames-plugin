import os
import logging
from typing import Tuple, Union, Optional
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

import fiftyone as fo

logger = logging.getLogger(__name__)
cv2.setNumThreads(1)

from suc_utils import evaluate_success_rate
from embedding_utils import propagatability_pre_label, propagatability_post_label
from labelprop_methods.siamese import PropagatorSiamFC
from labelprop_methods.swintrack import PropagatorSwinTrack
from labelprop_methods.sam2 import PropagatorSAM2


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


def propagate_annotations_pairwise(
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
    propagator = PropagatorSiamFC()
    # propagator = PropagatorSwinTrack()

    def process_sample(sample):
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

            propagator.register_source_frame(exemplar_frame, exemplar_detections)
            propagated_detections = propagator.propagate_to_target_frame(sample_frame)

            sample[output_annotation_field] = propagated_detections

            # If the sample already has an input annotation field, evaluate against it
            if evaluate_propagation and sample[input_annotation_field]:
                original_detections = sample[input_annotation_field]
                # TODO(neeraja): decouple the matching and the evaluation
                sample_score = evaluate_success_rate(original_detections, propagated_detections)
                logger.debug(f"Sample {sample.id} score: {sample_score}")
                return sample_score
        
        return None

    results = view.map_samples(process_sample, num_workers=1, save=True)
    scores = {sample_id: score for sample_id, score in results if score is not None}
    
    return scores


def propagate_annotations_sequential(
    view: Union[fo.core.dataset.Dataset, fo.core.view.DatasetView],
    exemplar_frame_field: str,
    input_annotation_field: str,
    output_annotation_field: str,
    evaluate_propagation: Optional[bool] = True,
    sort_field: Optional[str] = "frame_number",
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
    exemplar_propagators = {}
    
    def process_sample(sample):
        if sample[exemplar_frame_field]["is_exemplar"]:
            sample[output_annotation_field] = sample[input_annotation_field]
            return None
        elif len(sample[exemplar_frame_field]["exemplar_assignment"]) > 0:
            exemplar_frame_ids = sample[exemplar_frame_field]["exemplar_assignment"]
            # TODO(neeraja): handle multiple exemplar frames for the same sample
            exemplar_sample = view[exemplar_frame_ids[0]]

            if exemplar_sample.id not in exemplar_propagators:
                # exemplar_propagator = PropagatorSiamFC()
                exemplar_propagator = PropagatorSwinTrack()

                exemplar_frame = cv2.imread(exemplar_sample.filepath)
                exemplar_detections = exemplar_sample[input_annotation_field]
                exemplar_propagator.register_source_frame(exemplar_frame, exemplar_detections)
                exemplar_propagators[exemplar_sample.id] = exemplar_propagator
            else:
                exemplar_propagator = exemplar_propagators[exemplar_sample.id]

            sample_frame = cv2.imread(sample.filepath)
            propagated_detections = exemplar_propagator.propagate_to_target_frame(sample_frame)
        else:
            propagated_detections = fo.Detections(detections=[])

        sample[output_annotation_field] = propagated_detections

        # If the sample already has an input annotation field, evaluate against it
        if evaluate_propagation and sample[input_annotation_field]:
            original_detections = sample[input_annotation_field]
            # TODO(neeraja): decouple the matching and the evaluation
            sample_score = evaluate_success_rate(original_detections, propagated_detections)
            logger.debug(f"Sample {sample.id} score: {sample_score}")
            return sample_score
        
        return None

    if view.has_field(sort_field):
        results = view.sort_by(sort_field).map_samples(process_sample, num_workers=1, save=True)
    else:
        results = view.map_samples(process_sample, num_workers=1, save=True)
    scores = {sample_id: score for sample_id, score in results if score is not None}
    
    return scores


def propagate_annotations_sam2(
    view: Union[fo.core.dataset.Dataset, fo.core.view.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    evaluate_propagation: Optional[bool] = True,
    sort_field: Optional[str] = "frame_number",
) -> dict[str, float]:
    """
    Propagate annotations from exemplar frames (containing labels in input_annotation_field) to all the frames.
    Args:
        view: The view to propagate annotations from
        input_annotation_field: The field name of the annotation to copy from the exemplar frame field
        output_annotation_field: The field name of the annotation to save to the target frame
        evaluate_propagation: Whether to evaluate the propagation against
                              the input annotation field present in the propagation targets.
    """

    # Set up the propagator
    propagator = PropagatorSAM2()

    if view.has_field(sort_field):
        image_path_list = view.sort_by(sort_field).values("filepath")
    else:
        image_path_list = view.values("filepath")
    propagator.initialize(image_path_list)

    # Register all frames
    def register_sample(sample):
        if sample[input_annotation_field]:
            propagator.register_source_frame(sample.filepath, sample[input_annotation_field])

    _ = list(view.map_samples(register_sample, num_workers=1))

    # Propagate
    propagator.propagate_to_all_frames()

    # Process each sample
    def process_sample(sample):
        if sample[input_annotation_field]:
            sample[output_annotation_field] = sample[input_annotation_field]
            return None
        else:
            sample_filepath = sample.filepath
            propagated_detections = propagator.propagate_to_target_frame(sample_filepath)

        sample[output_annotation_field] = propagated_detections

        # If the sample already has an input annotation field, evaluate against it
        if evaluate_propagation and sample[input_annotation_field]:
            original_detections = sample[input_annotation_field]
            # TODO(neeraja): decouple the matching and the evaluation
            sample_score = evaluate_success_rate(original_detections, propagated_detections)
            logger.debug(f"Sample {sample.id} score: {sample_score}")
            return sample_score
        
        return None

    if view.has_field(sort_field):
        results = view.sort_by(sort_field).map_samples(process_sample, num_workers=1, save=True)
    else:
        results = view.map_samples(process_sample, num_workers=1, save=True)
    scores = {sample_id: score for sample_id, score in results if score is not None}
    
    return scores



def estimate_propagatability(
    view: Union[fo.core.dataset.Dataset, fo.core.view.DatasetView],
    exemplar_frame_field: str,
    input_annotation_field: str,
) -> dict[str, float]:
    """
    Estimate the propagatability of the annotations from the exemplar frames to the target frames.
    Args:
        view: The view to propagate annotations from
        exemplar_frame_field: The field name in which the exemplar frame assignments are stored
        input_annotation_field: The field name of the annotation to copy from the exemplar frame field
    """
    def process_sample(sample):
        if (not sample[exemplar_frame_field]["is_exemplar"]) and \
        (len(sample[exemplar_frame_field]["exemplar_assignment"]) > 0):
            exemplar_frame_ids = sample[exemplar_frame_field]["exemplar_assignment"]

            # TODO(neeraja): handle multiple exemplar frames for the same sample
            exemplar_sample = view[exemplar_frame_ids[0]]
            exemplar_frame = cv2.imread(exemplar_sample.filepath)
            exemplar_detections = exemplar_sample[input_annotation_field]
            sample_frame = cv2.imread(sample.filepath)

            propagatability_score = propagatability_pre_label(exemplar_frame, sample_frame)
            # propagatability_score = propagatability_post_label(exemplar_frame, sample_frame, exemplar_detections)
            return propagatability_score
        
        return None

    results = view.map_samples(process_sample, num_workers=1)
    scores = {sample_id: score for sample_id, score in results if score is not None}
    
    return scores
