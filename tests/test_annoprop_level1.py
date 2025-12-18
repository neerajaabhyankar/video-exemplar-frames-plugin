import fiftyone as fo
import fiftyone.operators as foo

dataset = fo.load_dataset("basketball_frames")
dataset_level1 = dataset.load_saved_view("spinning_part1")

if "exemplar_test_run_key" in dataset.list_runs():
    dataset.delete_run("exemplar_test_run_key")

ctx1 = {
    "dataset": dataset,
    "view": dataset_level1,
    "params": {
        "exemplar_frame_field": "exemplar_test",
        "max_fraction_exemplars": 0.1,
        "exemplar_run_key": "exemplar_test_run_key",
    },
}

exemplar_result = foo.execute_operator(
    "@neerajaabhyankar/video-exemplar-frames-plugin/extract_exemplar_frames",
    ctx1
)
# exemplar_assignments = exemplar_result.result["exemplar_assignments"]
print(exemplar_result.result["message"])

# TODO(neeraja): add option to pass a custom assignment to annoprop

ctx2 = {
    "dataset": dataset,
    "view": dataset_level1,
    "params": {
        "exemplar_run_key": exemplar_result.result["run_key"],
        "annotation_field": "sam",
    },
}

# TODO(neeraja): add an inbuilt evaluation flag to this --> record accuracy in the result

anno_prop_result = foo.execute_operator(
    "@neerajaabhyankar/video-exemplar-frames-plugin/propagate_annotations_from_exemplars",
    ctx2
)
# print(anno_prop_result.result["message"])

session = fo.launch_app(dataset_level1)
session.wait()
