import os
import json
import fiftyone as fo


# create a new dataset from the files in this directory
dataset = fo.Dataset.from_dir(
    dataset_dir="/Users/neeraja/video-exemplar-frames-plugin/basketball_frames",
    dataset_type=fo.types.ImageDirectory,
    name="basketball_frames",
    overwrite=True,
)


dataset.set_values
# Extract frame_number from filenames (format: frame_XXXXXX.jpg)
def extract_frame_number(sample):
    filename = os.path.basename(sample.filepath)
    frame_num_str = filename.replace("frame_", "").replace(".jpg", "")
    sample["frame_number"] = int(frame_num_str)

# TODO(neeraja): use map_samples instead
for sample in dataset:
    extract_frame_number(sample)
    sample.save()


# Add annotations to
anno_key = "ha_test_1"

labels_file = "/Users/neeraja/video-exemplar-frames-plugin/basketball_frames/partial_annotations.json"
with open(labels_file, "r") as f:
    labels = json.load(f)
    frame_map = labels["labels"]  # {"frame_002728": [...], ...}

def add_annotations(sample):
    key = f"frame_{sample.frame_number:06d}"
    objs = frame_map.get(key)
    if objs:
        sample[anno_key] = fo.Detections(
        detections=[
            fo.Detection(label=obj["label"], bounding_box=obj["bounding_box"])
            for obj in objs
        ]
    )

# TODO(neeraja): use map_samples instead
for sample in dataset:
    add_annotations(sample)
    sample.save()

dataset.persistent = True
dataset.save()

assert "basketball_frames" in fo.list_datasets()


"""
1. For computing embeddings, run `python scripts/embeddings.py` (rerun if you get a segfault)
2. In the UI, create views based on the embeddings
"""