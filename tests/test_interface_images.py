import pytest
import fiftyone as fo
import fiftyone.operators as foo
import numpy as np


VIEW_NAME = "spinning_part1"


@pytest.fixture
def clean_dataset_fixture():
    dataset = fo.load_dataset("basketball_frames")
    return dataset


@pytest.fixture
def dataset_fixture():
    dataset = fo.load_dataset("basketball_frames")
    return dataset


@pytest.mark.dependency()
def test_extract_exemplar_frames(clean_dataset_fixture):
    ctx1 = {
        "dataset": clean_dataset_fixture,
        "view": clean_dataset_fixture.load_saved_view(VIEW_NAME),
        "params": {
            "method": "zcore:embeddings_hausdorff_nbd_mds_8",
            "exemplar_frame_field": "exemplar_test",
            "max_fraction_exemplars": 0.1,
        },
    }

    exemplar_result = foo.execute_operator(
        "@neerajaabhyankar/video-exemplar-frames-plugin/extract_exemplar_frames",
        ctx1
    )
    print(exemplar_result.result["message"])
    assert "exemplar_test.is_exemplar" in clean_dataset_fixture.get_field_schema(flat=True)


@pytest.mark.dependency(depends=["test_extract_exemplar_frames"])
def test_propagate_labels_assigned(dataset_fixture):
    ctx2 = {
        "dataset": dataset_fixture,
        "view": dataset_fixture.load_saved_view(VIEW_NAME),
        "params": {
            "exemplar_frame_field": "exemplar_test",
            "input_annotation_field": "ha_test_1",
            "output_annotation_field": "ha_test_1_propagated",
        },
    }

    anno_prop_result = foo.execute_operator(
        "@neerajaabhyankar/video-exemplar-frames-plugin/propagate_labels_from_assigned_exemplars",
        ctx2
    )
    print(anno_prop_result.result["message"])
    for sample_id, score in anno_prop_result.result["propagation_score"].items():
        print(f"Sample {sample_id} score: {score}")
    if len(anno_prop_result.result["propagation_score"]) > 0:
        print(f"Average propagation score: {np.mean(list(anno_prop_result.result['propagation_score'].values()))}")
