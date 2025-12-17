import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz

dataset = fo.load_dataset("basketball_frames")
single_sample_dataset = dataset[["6939cb19464fecaa395fb442"]]

assert len(single_sample_dataset) == 1
sam_sample = single_sample_dataset.first()
old_mask = np.array(sam_sample.sam.detections[0].mask)
print(np.sum(old_mask))

seg_model = foz.load_zoo_model("segment-anything-vitb-torch")

# perform zeroshot inference
single_sample_dataset.apply_model(
    seg_model,
    label_field="sam", prompt_field="sam"
)
sam_sample = single_sample_dataset.first()
new_mask = np.array(sam_sample.sam.detections[0].mask)
print(np.sum(new_mask))
