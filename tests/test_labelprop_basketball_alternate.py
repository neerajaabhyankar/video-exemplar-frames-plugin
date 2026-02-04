import pytest
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

import fiftyone as fo
import fiftyone.operators as foo

sys.path.insert(0, str(Path(__file__).parent.parent))
from annoprop import (
    propagate_annotations_pairwise,
    propagate_annotations_sequential,
    propagate_annotations_sam2, 
    estimate_propagatability
)
from utils import evaluate_success_rate


VIEW_NAME = "underbasket_reverse_layup"
# LIMITATION: when the person goes out of the frame, we get a false positive detection


@pytest.fixture
def dataset_slice():
    dataset = fo.load_dataset("basketball_frames")
    dataset_slice = dataset.load_saved_view(VIEW_NAME).limit(50)
    return dataset_slice


@pytest.fixture
def exemplar_assigned_dataset_slice(dataset_slice):
    if "exemplar_test" in dataset_slice._dataset.get_field_schema():
        dataset_slice._dataset.delete_sample_field("exemplar_test")

    exemplar_id = dataset_slice.first().id
    for ii, sample in enumerate(dataset_slice.sort_by("frame_number")):
        if ii % 2 == 0:
            is_exemplar = True
            exemplar_id = sample.id
        else:
            is_exemplar = False
        sample["exemplar_test"] = {
            "is_exemplar": is_exemplar,
            "exemplar_assignment": [exemplar_id] if not is_exemplar else []
        }
        sample.save()
    
    return dataset_slice


@pytest.fixture
def partially_labeled_dataset_slice(dataset_slice):
    if "human_labels_test" in dataset_slice._dataset.get_field_schema():
        dataset_slice._dataset.delete_sample_field("human_labels_test")
        dataset_slice._dataset.add_sample_field(
            "human_labels_test",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )

    for ii, sample in enumerate(dataset_slice):
        if ii % 2 == 0:
            sample["human_labels_test"] = sample["ha_test_1"]
            # # randomly add only one of the two labels
            # # LIMITATION: SAM2 errors out if there's no continuity in the labels?!
            # if sample["ha_test_1"] is not None:
            #     li = np.random.randint(len(sample["ha_test_1"].detections))
            #     sample["human_labels_test"] = fo.Detections(
            #         detections=[sample["ha_test_1"].detections[li]]
            #     )
            sample.save()
    
    return dataset_slice


def test_propagation(exemplar_assigned_dataset_slice):
    score = propagate_annotations_pairwise(
        exemplar_assigned_dataset_slice,
        exemplar_frame_field="exemplar_test", 
        input_annotation_field="ha_test_1",
        output_annotation_field="ha_test_1_propagated",
    )

    for sample_id, sample_score in score.items():
        print(f"Sample {sample_id} score: {sample_score}")
    
    print(f"Average propagation score: {np.mean(list(score.values()))}")
    
    assert np.mean(list(score.values())) > 0.33
    # session = fo.launch_app(exemplar_assigned_dataset_slice)
    # session.wait()


def test_propagatability(exemplar_assigned_dataset_slice):
    score_estimate = estimate_propagatability(
        exemplar_assigned_dataset_slice,
        exemplar_frame_field="exemplar_test",
        input_annotation_field="ha_test_1",
    )
    score = propagate_annotations_pairwise(
        exemplar_assigned_dataset_slice,
        exemplar_frame_field="exemplar_test",
        input_annotation_field="ha_test_1",
        output_annotation_field="ha_test_1_propagated",
    )
    # Align dicts on common sample IDs
    common_ids = sorted(set(score_estimate.keys()) & set(score.keys()))
    est_array = np.array([score_estimate[_id] for _id in common_ids], dtype=float)
    true_array = np.array([score[_id] for _id in common_ids], dtype=float)

    # Correlations (no assertions for now)
    spearman_corr, spearman_p = spearmanr(est_array, true_array)
    pearson_corr, pearson_p = pearsonr(est_array, true_array)

    print("Correlation between estimated and true propagation scores:")
    print(f"  Spearman: {spearman_corr:.3f} (p={spearman_p:.4f})")
    print(f"  Pearson:  {pearson_corr:.3f} (p={pearson_p:.4f})")

    # # Scatter plot: estimate vs true
    # plt.figure(figsize=(5, 4))
    # plt.scatter(est_array, true_array, alpha=0.6, c="steelblue")
    # plt.xlabel("Estimated Propagatability")
    # plt.ylabel("True Propagation Score")
    # plt.title(f"Estimate vs True (Spearman: {spearman_corr:.3f})")
    # plt.grid(True, alpha=0.7)
    # plt.tight_layout()
    # plt.show()


def test_propagate_labels_unassigned(partially_labeled_dataset_slice):
    GROUND_TRUTH_FIELD = "ha_test_1"
    INPUT_ANNOTATION_FIELD = "human_labels_test"
    OUTPUT_ANNOTATION_FIELD = "human_labels_test_propagated"
    ctx2 = {
        "dataset": partially_labeled_dataset_slice._dataset,
        "view": partially_labeled_dataset_slice,
        "params": {
            "input_annotation_field": INPUT_ANNOTATION_FIELD,
            "output_annotation_field": OUTPUT_ANNOTATION_FIELD,
        },
    }

    anno_prop_result = foo.execute_operator(
        "@neerajaabhyankar/video-exemplar-frames-plugin/propagate_labels_from_exemplars",
        ctx2
    )
    print(anno_prop_result.result["message"])

    scores = []
    for sample in partially_labeled_dataset_slice:
        if sample[INPUT_ANNOTATION_FIELD] is not None:
            continue
        gt_detections = sample[GROUND_TRUTH_FIELD]
        propagated_detections = sample[OUTPUT_ANNOTATION_FIELD]
        sample_score = evaluate_success_rate(gt_detections, propagated_detections)
        scores.append(sample_score)
        print(f"Sample {sample.id} score: {sample_score}")
    print(f"Average propagation score: {np.mean(scores)}")
    
    # assert np.mean(scores) > 0.7
    session = fo.launch_app(partially_labeled_dataset_slice)
    session.wait()