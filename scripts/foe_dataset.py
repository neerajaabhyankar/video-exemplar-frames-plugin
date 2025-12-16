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
    "ha_test_1": {
        "type": "detections",
        "classes": [
            "person",
            "basketball",
        ],
    },
}

# Note: this is not how the future API will look like
# make a new field
# new_field = fo.Detections(
#     target_field=anno_key,
#     schema=label_schema[anno_key],
# )
dataset.add_sample_field(
    anno_key,
    ftype=fof.EmbeddedDocumentField,
    embedded_doc_type=fol.Detections,
    schema=label_schema[anno_key],
)
dataset.save()