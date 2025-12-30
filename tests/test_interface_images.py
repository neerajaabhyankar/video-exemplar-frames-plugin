import pytest
import fiftyone as fo
import fiftyone.operators as foo


TEST_RUN_KEY = "exemplar_test_run_key"
TEST_RESULT_OBJECT = None


@pytest.fixture
def clean_dataset_fixture():
    dataset = fo.load_dataset("basketball_frames")
    if TEST_RUN_KEY in dataset.list_runs():
        dataset.delete_run(TEST_RUN_KEY)
    return dataset


@pytest.fixture
def dataset_fixture():
    dataset = fo.load_dataset("basketball_frames")
    return dataset


@pytest.fixture
def exemplar_result_object():
    """Fixture that returns the result from test_extract_exemplar_frames."""
    if TEST_RESULT_OBJECT is None:
        pytest.fail("test_extract_exemplar_frames must run first and store its result")
    return TEST_RESULT_OBJECT


@pytest.mark.dependency()
def test_extract_exemplar_frames(clean_dataset_fixture):
    global TEST_RESULT_OBJECT
    ctx1 = {
        "dataset": clean_dataset_fixture,
        "params": {
            "exemplar_frame_field": "exemplar_test",
            "max_fraction_exemplars": 0.1,
            "exemplar_run_key": TEST_RUN_KEY,
        },
    }

    exemplar_result = foo.execute_operator(
        "@neerajaabhyankar/video-exemplar-frames-plugin/extract_exemplar_frames",
        ctx1
    )
    # Store the result in the global variable for use in other tests
    TEST_RESULT_OBJECT = exemplar_result
    # exemplar_assignments = exemplar_result.result["exemplar_assignments"]
    print(exemplar_result.result["message"])


@pytest.mark.dependency(depends=["test_extract_exemplar_frames"])
def test_propagate_annotations(dataset_fixture, exemplar_result_object):
    ctx2 = {
        "dataset": dataset_fixture,
        "params": {
            "exemplar_run_key": exemplar_result_object.result["run_key"],
            "input_annotation_field": "sam",
            "output_annotation_field": "sam_propagated",
        },
    }

    anno_prop_result = foo.execute_operator(
        "@neerajaabhyankar/video-exemplar-frames-plugin/propagate_annotations_from_exemplars",
        ctx2
    )
    print(anno_prop_result.result["message"])
