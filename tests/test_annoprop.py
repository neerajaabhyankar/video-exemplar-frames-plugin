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

# Set the exemplar frame field and assignments
exemplar_assignments = {}
dataset.add_sample_field("exemplar_frame", fo.BooleanField)
exemplar_id = None
for ii, sample in enumerate(dataset_slice.sort_by("frame_number")):
    if ii % 2 == 0:
        sample["exemplar_frame"] = True
        exemplar_id = sample.id
    else:
        sample["exemplar_frame"] = False
    exemplar_assignments[sample.id] = [exemplar_id]
    sample.save()


score = propagate_annotations(
    dataset_slice,
	exemplar_frame_field="exemplar_frame", 
	input_annotation_field="ha_test_1",
	output_annotation_field="ha_test_1_propagated",
	exemplar_assignments=exemplar_assignments,
)

print(f"Score: {score}")

session = fo.launch_app(dataset_slice)
session.wait()
