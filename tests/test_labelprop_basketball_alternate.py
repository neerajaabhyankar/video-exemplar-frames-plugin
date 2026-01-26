import pytest
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

import fiftyone as fo
import fiftyone.operators as foo

sys.path.insert(0, str(Path(__file__).parent.parent))
from annoprop import propagate_annotations, estimate_propagatability


@pytest.fixture
def dataset_slice():
    dataset = fo.load_dataset("basketball_frames")
    dataset_slice = dataset.load_saved_view("side_top_layup")
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


def test_labelprop_basketball(exemplar_assigned_dataset_slice):
    score = propagate_annotations(
        exemplar_assigned_dataset_slice,
        exemplar_frame_field="exemplar_test", 
        input_annotation_field="ha_test_1",
        output_annotation_field="ha_test_1_propagated",
    )

    for sample_id, sample_score in score.items():
        print(f"Sample {sample_id} score: {sample_score}")
    
    assert np.mean(list(score.values())) > 0.4


def test_propagatability_basketball(exemplar_assigned_dataset_slice):
    score_estimate = estimate_propagatability(
        exemplar_assigned_dataset_slice,
        exemplar_frame_field="exemplar_test",
        input_annotation_field="ha_test_1",
    )
    score = propagate_annotations(
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

    # Scatter plot: estimate vs true
    plt.figure(figsize=(5, 4))
    plt.scatter(est_array, true_array, alpha=0.6, c="steelblue")
    plt.xlabel("Estimated Propagatability")
    plt.ylabel("True Propagation Score")
    plt.title(f"Estimate vs True (Spearman: {spearman_corr:.3f})")
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.show()