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


"""
Inspect Embeddings
"""


import fiftyone as fo
dataset = fo.load_dataset("basketball_frames")
dataset_slice = dataset.load_saved_view("side_top_layup")

import cv2
import torch
import numpy as np
from annoprop_algos import setup_siamfc

tracker = setup_siamfc()
tracker.net.eval()

all_embeddings = []

for sample in dataset_slice:
    img = cv2.imread(sample.filepath)
    # Convert to torch tensor and process through backbone (similar to siamfc.py:159-166)
    # but without cropping - send entire image
    x = torch.from_numpy(img).to(
        tracker.device).permute(2, 0, 1).unsqueeze(0).float()
    embedding = tracker.net.backbone(x)
    embedding = embedding.detach().cpu().numpy().squeeze(0)
    # embedding is of shape (256, 35, 70)
    # with the latter two being spatial dimensions
    all_embeddings.append(embedding)


all_embeddings = np.array(all_embeddings)


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

""" center only """
center_embedding = all_embeddings[:, :, 17, 35]

# import umap
# umap_model = umap.UMAP(n_neighbors=3, min_dist=0.1, n_components=2)
# umap_embedding = umap_model.fit_transform(center_embedding)
# plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1])
# plt.show()

tsne = TSNE(n_components=2, init='pca', random_state=501)
tsne_embedding = tsne.fit_transform(center_embedding)
plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1])
plt.show()

""" a 10x10 block around the center """

#TODO(neeraja): verify/fix


# Extract 10x10 block around center (17, 35)
# Using 12:22 and 30:40 to get exactly 10 elements in each dimension
block_embeddings = all_embeddings[:, :, 12:22, 30:40]  # Shape: (45, 256, 10, 10)

# Flatten to (4500, 256): (num_samples * 10 * 10, 256)
num_samples = block_embeddings.shape[0]
flattened_embeddings = block_embeddings.transpose(0, 2, 3, 1).reshape(-1, 256)  # Shape: (4500, 256)

# Create 2D colormap for the 10x10 positions
import colorstamps
# Generate grid coordinates for the 10x10 block (normalized to -1 to 1)
x_coords = np.linspace(-1, 1, 10)
y_coords = np.linspace(-1, 1, 10)
X, Y = np.meshgrid(x_coords, y_coords)

# Apply the colormap to map 2D positions to colors
rgb, stamp = colorstamps.apply_stamp(X, Y, 'peak',
                                     vmin_0=-1, vmax_0=1,
                                     vmin_1=-1, vmax_1=1)
# rgb has shape (10, 10, 3)
flattened_colormap = rgb.reshape(-1, 3)  # Shape: (100, 3)

# Repeat the colormap for each sample to match the flattened embeddings
colors = np.tile(flattened_colormap, (num_samples, 1))  # Shape: (4500, 3)

# Apply t-SNE
tsne_block = TSNE(n_components=2, init='pca', random_state=501)
tsne_embedding_block = tsne_block.fit_transform(flattened_embeddings)

# Plot with colors corresponding to 2D position in the 10x10 block
plt.figure(figsize=(10, 8))
plt.scatter(tsne_embedding_block[:, 0], tsne_embedding_block[:, 1], c=colors, s=5, alpha=0.6)
plt.title('t-SNE Visualization of 10x10 Block Embeddings (colored by spatial position)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()