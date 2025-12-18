"""
Run this script in the foe environment
"""

import os
from tqdm import tqdm
import numpy as np

import fiftyone as fo
import fiftyone.core.storage as fos
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol


"""
Init
"""
# this dataset was initialized via the UI
dataset = fo.load_dataset("basketball_frames_ha")

"""
Add samples to the dataset
"""
IMAGE_DIR = "/var/folders/r0/jhwzxbr9611_fgf73jbtdxbc0000gn/T/frames_rxotnotj/"
# dataset.add_dir(dataset_type=fo.types.COCODetectionDataset, data_path=IMAGE_DIR)
cloud_dir = "gs://voxel51-test/ml/custom-datasets/basketball_frames/"

# add a progress bar
for file in tqdm(os.listdir(IMAGE_DIR)):
    if file.endswith(".jpg"):
        sample = fo.Sample(
            filepath=os.path.join(IMAGE_DIR, file),
            frame_number=int(file.split("_")[1].split(".")[0]),
        )
        dataset.add_sample(sample)

fos.upload_media(dataset, cloud_dir, update_filepaths=True, overwrite=False)
dataset.persistent = True
dataset.save()

# to delete all samples from the dataset
# dataset.delete_samples(samples_or_ids=dataset.values("id"),)

"""
Embeddings
"""

# load embeddings for the dataset
embs = np.load("/Users/neerajaabhyankar/Downloads/basketball_frames_embeddings.npy")
emb_list = [np.array(emb) for emb in embs]

sorted_view = dataset.sort_by("frame_number")
for sample, embedding in tqdm(zip(sorted_view, emb_list), total=len(sorted_view)):
    sample["embedding"] = embedding
    sample.save()
dataset.save()

# compute visualization -- can also be done in the UI

"""
Annotations
"""

anno_key = "ha_test_1"
label_schema = {
    "type": "detections",
    "classes": [
        "person",
        "basketball",
    ],
}

dataset.add_sample_field(
    anno_key,
    fo.EmbeddedDocumentField,
    embedded_doc_type=fol.Detections,
    schema=label_schema,
)
dataset.save()

"""
Download annotations as labels in FiftyOne Image Detections format
Run the following in the fo environment
"""

import json
import fiftyone as fo
dataset = fo.load_dataset("basketball_frames")

labels_file = "/Users/neeraja/Downloads/basketball_annotations_phase_2_file-QYkYA5QH.json"

with open(labels_file, "r") as f:
    labels = json.load(f)

# labels["labels"].items() have
# key = "frame_002728"
# value = [{'label': 'basketball',
#   'bounding_box': [0.784967648474526,
#    0.6373296800947867,
#    0.026721249259478674,
#    0.04639983214849921]},
#  {'label': 'person',
#   'bounding_box': [0.8339323070941943,
#    0.46314672195892576,
#    0.1143098804058057,
#    0.22550108609794628]}]
# add these labels to the dataset

# TODO(neeraja): find a cleaner alternative to this

anno_key = "ha_test_1"  # your label field
frame_map = labels["labels"]  # {"frame_002728": [...], ...}

def to_detections(objs):
    return fo.Detections(
        detections=[
            fo.Detection(label=obj["label"], bounding_box=obj["bounding_box"])
            for obj in objs
        ]
    )

for sample in dataset:
    key = f"frame_{sample.frame_number:06d}"
    objs = frame_map.get(key)
    if objs:
        sample[anno_key] = to_detections(objs)
        sample.save()

dataset.save()