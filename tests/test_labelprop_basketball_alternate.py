import pytest
import sys
from pathlib import Path
import numpy as np

import fiftyone as fo
import fiftyone.operators as foo

sys.path.insert(0, str(Path(__file__).parent.parent))
from annoprop import propagate_annotations


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
