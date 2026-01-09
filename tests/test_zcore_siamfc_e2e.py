""" Imports """

import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.brain as fob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from annoprop import propagate_annotations
from annoprop_algos import setup_siamfc


""" Load dataset """

dataset = fo.load_dataset("basketball_frames")
dataset_slice = dataset.load_saved_view("spinning_part1")
# dataset_slice = dataset.load_saved_view("side_top_layup")
# dataset_slice = dataset.load_saved_view("underbasket_reverse_layup")
print(len(dataset_slice))


""" Compute SIAMFC embeddings """

tracker = setup_siamfc()
_ = tracker.net.eval()
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

all_embeddings = np.array(all_embeddings)  # Shape: (45, 256, 35, 70)
                                           # where 45 is the number of samples

print("SiamFC embeddings computed")


""" Compute Hausdorff distances """

# # # TODO(neeraja): move to asymmetric dtances

NN, DD, HH, WW = all_embeddings.shape
PATCH_NBD = max(HH, WW) // 10

def hausdorff_distance_between_images_nbdbased(img1, img2):
    min_deltas = []
    for hh in range(HH):
        for ww in range(WW):
            patch2 = img2[:, hh, ww]
            img1_neighborhood = img1[:, max(0, hh-PATCH_NBD):min(img1.shape[1], hh+PATCH_NBD), max(0, ww-PATCH_NBD):min(img1.shape[2], ww+PATCH_NBD)]
            patch_deltas = np.linalg.norm(img1_neighborhood - patch2[:, np.newaxis, np.newaxis], axis=0)
            min_deltas.append(np.min(patch_deltas))
    return np.max(min_deltas)

hausdorff_matrix = np.zeros((NN, NN))
for ii in range(NN):
    for jj in range(ii+1, NN):
        hausdorff_matrix[ii, jj] = hausdorff_distance_between_images_nbdbased(all_embeddings[ii], all_embeddings[jj])

# fill up the lower triangle
hausdorff_matrix = np.triu(hausdorff_matrix) + np.triu(hausdorff_matrix, 1).T

print("Hausdorff distances computed")


""" Compute MDS embeddings from the Hausdorff distance matrix """

MDS_DIM = 8         # target embedding dimension
MDS_EIG_TOL = 1e-8  # eigenvalue threshold

def compute_classical_mds_embedding(D, dim=MDS_DIM, eig_tol=MDS_EIG_TOL):
    """
    Classical MDS embedding from a distance matrix.

    Args:
        D (np.ndarray): NxN symmetric distance matrix
        dim (int): target embedding dimension
        eig_tol (float): eigenvalue threshold

    Returns:
        X (np.ndarray): Nxk embedding (k <= dim)
        eigvals (np.ndarray): eigenvalues used
    """
    D = np.asarray(D)
    N = D.shape[0]

    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ (D ** 2) @ J

    eigvals, eigvecs = np.linalg.eigh(B)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Keep strictly positive eigenvalues
    pos = eigvals > eig_tol
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]

    # Truncate to target dim
    k = min(dim, eigvals.shape[0])
    eigvals = eigvals[:k]
    eigvecs = eigvecs[:, :k]

    X = eigvecs @ np.diag(np.sqrt(eigvals))
    return X, eigvals

mds_embedding, _ = compute_classical_mds_embedding(hausdorff_matrix)

print("MDS embeddings computed")

embedding_field_name = "embeddings_hausdorff_nbd_mds_8"
dataset_slice._dataset.add_sample_field(embedding_field_name, fo.VectorField, shape=(MDS_DIM,))

for sample, mds_emb in zip(dataset_slice, mds_embedding):
    sample[embedding_field_name] = mds_emb
    sample.save()


# """ Compute ZCore scores from the MDS embeddings """

# embedding_field_name = "embeddings_hausdorff_nbd_mds_8"
zcore_score_field_name = "zcore_score_hausdorff_nbd_mds_8"

ctx = {
    "dataset": dataset,
    "view": dataset_slice,
    "params": {
        "embeddings": embedding_field_name,
        "zcore_score_field": zcore_score_field_name,
    },
}
foo.execute_operator("@51labs/zero-shot-coreset-selection/compute_zcore_score", ctx)

print("ZCore scores computed")


""" Visualize """

# session = fo.launch_app(dataset_slice)
# session.wait()


""" Select """

SELECT_FRACTION = 0.2
SELECT_NUMBER = int(len(dataset_slice) * SELECT_FRACTION)

uniform_samples = dataset_slice.sort_by("frame_number")[:SELECT_NUMBER]
random_samples = dataset_slice.take(SELECT_NUMBER)

zcore_hausdorff_samples = dataset_slice.sort_by(
    zcore_score_field_name, reverse=True
)[:SELECT_NUMBER]

zcore_clip_samples = dataset_slice.sort_by(
    "zcore_score_clip", reverse=True
)[:SELECT_NUMBER]


""" Assign to Exemplar """

def assign_to_nearest_neighbor_exemplar(
    all_samples,
    exemplar_samples,
    embedding_field,
    exemplar_indicator_field,
) -> dict:
    """
    Assign each sample to the nearest neighbor among the exemplars.
    The NN is defined by the embedding in the provided embedding_field.
    """
    if exemplar_indicator_field not in all_samples.first().field_names:
        all_samples._dataset.add_sample_field(exemplar_indicator_field, fo.BooleanField)

    exemplar_assignments = {}
    exemplar_ids = exemplar_samples.values("id")
    exemplar_embeddings = exemplar_samples.values(embedding_field)
    for sample in all_samples:
        nnbr_index = np.argmin(
            np.linalg.norm(
                exemplar_embeddings - sample[embedding_field], axis=1
            )
        )
        nnbr_id = exemplar_ids[nnbr_index]
        # Assign the sample to the nearest neighbor
        if nnbr_id == sample.id:
            sample[exemplar_indicator_field] = True
        else:
            sample[exemplar_indicator_field] = False
        exemplar_assignments[sample.id] = [nnbr_id]
    
    all_samples.save()
    return all_samples, exemplar_assignments


dataset_slice, uniform_assignments = assign_to_nearest_neighbor_exemplar(
    dataset_slice,
    uniform_samples,
    embedding_field_name,
    "exemplar_uniform",
)

dataset_slice, random_assignments = assign_to_nearest_neighbor_exemplar(
    dataset_slice,
    random_samples,
    embedding_field_name,
    "exemplar_random",
)

dataset_slice, zcore_hausdorff_assignments = assign_to_nearest_neighbor_exemplar(
    dataset_slice,
    zcore_hausdorff_samples,
    embedding_field_name,
    "exemplar_zcore_hausdorff",
)

dataset_slice, zcore_clip_assignments = assign_to_nearest_neighbor_exemplar(
    dataset_slice,
    zcore_clip_samples,
    embedding_field_name,
    "exemplar_zcore_clip",
)

""" Propagate + Evaluate """

uniform_score = propagate_annotations(
    dataset_slice,
    exemplar_frame_field="exemplar_uniform",
    input_annotation_field="ha_test_1",
    output_annotation_field="ha_test_1_propagated",
    exemplar_assignments=uniform_assignments,
)
print(f"Uniform score: {uniform_score}")

random_score = propagate_annotations(
    dataset_slice,
    exemplar_frame_field="exemplar_random",
    input_annotation_field="ha_test_1",
    output_annotation_field="ha_test_1_propagated",
    exemplar_assignments=random_assignments,
)
print(f"Random score: {random_score}")

zcore_clip_score = propagate_annotations(
    dataset_slice,
    exemplar_frame_field="exemplar_zcore_clip",
    input_annotation_field="ha_test_1",
    output_annotation_field="ha_test_1_propagated",
    exemplar_assignments=zcore_clip_assignments,
)
print(f"ZCore Clip score: {zcore_clip_score}")

zcore_hausdorff_score = propagate_annotations(
    dataset_slice,
    exemplar_frame_field="exemplar_zcore_hausdorff",
    input_annotation_field="ha_test_1",
    output_annotation_field="ha_test_1_propagated",
    exemplar_assignments=zcore_hausdorff_assignments,
)
print(f"ZCore Hausdorff score: {zcore_hausdorff_score}")
