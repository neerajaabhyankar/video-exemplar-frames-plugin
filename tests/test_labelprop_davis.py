import pytest
import sys
from pathlib import Path
import numpy as np

import fiftyone as fo
import fiftyone.operators as foo

sys.path.insert(0, str(Path(__file__).parent.parent))
from annoprop import propagate_annotations


@pytest.fixture
def dataset():
    dataset = fo.load_dataset("davis-2017")
    dataset = dataset.match_tags("val")
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


def test_labelprop_davis(exemplar_assigned_dataset):
    score = propagate_annotations(
        exemplar_assigned_dataset,
        exemplar_frame_field="exemplar_first_frame", 
        input_annotation_field="ground_truth",
        output_annotation_field="ground_truth_propagated",
    )

    for sample_id, sample_score in score.items():
        print(f"Sample {sample_id} score: {sample_score}")
    
    print(f"Average propagation score: {np.mean(list(score.values()))}")
    assert np.mean(list(score.values())) > 0.33
