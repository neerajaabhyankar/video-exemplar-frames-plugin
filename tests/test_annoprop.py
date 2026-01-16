import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fiftyone as fo
import fiftyone.operators as foo

from annoprop import propagate_annotations


dataset = fo.load_dataset("basketball_frames")
# dataset_slice = dataset.load_saved_view("spinning_part1")
dataset_slice = dataset.load_saved_view("side_top_layup")
# dataset_slice = dataset.load_saved_view("underbasket_reverse_layup")


"""
For now: pick every alternate frame as an exemplar

Later: "Cross-Propagation"
Choose each frame as an exemplar turn-by-turn
And record the accuracy of the propagation from it
"""

if "exemplar_test" in dataset_slice._dataset.get_field_schema():
    dataset_slice._dataset.delete_sample_field("exemplar_test")

# Set the exemplar frame field and assignments
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


score = propagate_annotations(
    dataset_slice,
	exemplar_frame_field="exemplar_test", 
	input_annotation_field="ha_test_1",
	output_annotation_field="ha_test_1_propagated",
)

print(f"Score: {score}")

session = fo.launch_app(dataset_slice)
session.wait()
