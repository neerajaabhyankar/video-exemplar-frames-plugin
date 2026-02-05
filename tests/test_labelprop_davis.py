import pytest
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo

sys.path.insert(0, str(Path(__file__).parent.parent))
from annoprop import (
    propagate_annotations_pairwise,
    propagate_annotations_sequential,
    propagate_annotations_sam2,
    estimate_propagatability
)
from utils import evaluate


@pytest.fixture
def dataset():
    dataset = foz.load_zoo_dataset("https://github.com/voxel51/davis-2017", split="validation", format="image")
    SELECT_SEQUENCES = ["bike-packing"]
    # SELECT_SEQUENCES = ["blackswan", "breakdance", "india"]  # TODO(neeraja): get to work for multiple sequences
    dataset = dataset.match_tags(SELECT_SEQUENCES)
    return dataset


@pytest.fixture
def exemplar_assigned_dataset(dataset):
    if "exemplar_first_frame" in dataset._dataset.get_field_schema():
        dataset._dataset.delete_sample_field("exemplar_first_frame")
    
    sequences = dataset.distinct("tags")
    sequences.remove("val")

    for seq in sequences:
        dataset_slice = dataset.match_tags(seq).sort_by("frame_number")
        exemplar_id = dataset_slice.first().id
        for ii, sample in enumerate(dataset_slice):
            sample["exemplar_first_frame"] = {
                "is_exemplar": ii == 0,
                "exemplar_assignment": [exemplar_id] if ii != 0 else []
            }
            sample.save()
    
    return dataset


@pytest.fixture
def partially_labeled_dataset(dataset):
    if "human_labels_test" in dataset._dataset.get_field_schema():
        dataset._dataset.delete_sample_field("human_labels_test")
        dataset._dataset.add_sample_field(
            "human_labels_test",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )
    
    sequences = dataset.distinct("tags")
    sequences.remove("val")

    new_frame_number = 0
    for seq in sequences:
        dataset_slice = dataset.match_tags(seq).sort_by("frame_number")
        dataset_slice.set_values(
            "new_frame_number",
            [new_frame_number + ii for ii in range(len(dataset_slice))]
        )
        new_frame_number += len(dataset_slice)
        exemplar_sample = dataset_slice.first()
        exemplar_sample["human_labels_test"] = exemplar_sample["ground_truth"]
        exemplar_sample.save()
    
    return dataset


def test_propagation(exemplar_assigned_dataset):
    score = propagate_annotations_pairwise(
        exemplar_assigned_dataset,
        exemplar_frame_field="exemplar_first_frame", 
        input_annotation_field="ground_truth",
        output_annotation_field="ground_truth_propagated",
    )

    for sample_id, sample_score in score.items():
        print(f"Sample {sample_id} score: {sample_score}")
    
    print(f"Average propagation score: {np.mean(list(score.values()))}")
    
    assert np.mean(list(score.values())) > 0.33
    # session = fo.launch_app(exemplar_assigned_dataset)
    # session.wait()


def test_propagation_sam2(partially_labeled_dataset):
    score = propagate_annotations_sam2(
        partially_labeled_dataset,
        input_annotation_field="human_labels_test",
        output_annotation_field="human_labels_test_propagated",
        sort_field="new_frame_number",
    )

    scores = []
    for sample in partially_labeled_dataset:
        gt_detections = sample["ground_truth"]
        propagated_detections = sample["human_labels_test_propagated"]
        sample_score = evaluate(gt_detections, propagated_detections)
        scores.append(sample_score)
        print(f"Sample {sample.id} score: {sample_score}")
    print(f"Average propagation score: {np.mean(scores)}")

    assert np.mean(scores) > 0.8
    # session = fo.launch_app(partially_labeled_dataset)
    # session.wait()


def test_propagatability(exemplar_assigned_dataset):
    score_estimate = estimate_propagatability(
        exemplar_assigned_dataset,
        exemplar_frame_field="exemplar_first_frame",
        input_annotation_field="ground_truth",
    )
    score = propagate_annotations_pairwise(
        exemplar_assigned_dataset,
        exemplar_frame_field="exemplar_first_frame", 
        input_annotation_field="ground_truth",
        output_annotation_field="ground_truth_propagated",
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

    # Scatter plot: estimate vs true
    plt.figure(figsize=(5, 4))
    plt.scatter(est_array, true_array, alpha=0.6, c="steelblue")
    plt.xlabel("Estimated Propagatability")
    plt.ylabel("True Propagation Score")
    plt.title(f"Estimate vs True (Spearman: {spearman_corr:.3f})")
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.show()
