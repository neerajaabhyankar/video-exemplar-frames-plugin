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

# tracker = setup_siamfc()
# _ = tracker.net.eval()
# all_embeddings = []

# for sample in dataset_slice:
#     img = cv2.imread(sample.filepath)
#     # Convert to torch tensor and process through backbone (similar to siamfc.py:159-166)
#     # but without cropping - send entire image
#     x = torch.from_numpy(img).to(
#         tracker.device).permute(2, 0, 1).unsqueeze(0).float()
#     embedding = tracker.net.backbone(x)
#     embedding = embedding.detach().cpu().numpy().squeeze(0)
#     # embedding is of shape (256, 35, 70)
#     # with the latter two being spatial dimensions
#     all_embeddings.append(embedding)

# all_embeddings = np.array(all_embeddings)  # Shape: (45, 256, 35, 70)
#                                            # where 45 is the number of samples

# print("SiamFC embeddings computed")


""" Compute Hausdorff distances """

# # # TODO(neeraja): move to asymmetric dtances

# NN, DD, HH, WW = all_embeddings.shape
# PATCH_NBD = max(HH, WW) // 10

# def hausdorff_distance_between_images_nbdbased(img1, img2):
#     min_deltas = []
#     for hh in range(HH):
#         for ww in range(WW):
#             patch2 = img2[:, hh, ww]
#             img1_neighborhood = img1[:, max(0, hh-PATCH_NBD):min(img1.shape[1], hh+PATCH_NBD), max(0, ww-PATCH_NBD):min(img1.shape[2], ww+PATCH_NBD)]
#             patch_deltas = np.linalg.norm(img1_neighborhood - patch2[:, np.newaxis, np.newaxis], axis=0)
#             min_deltas.append(np.min(patch_deltas))
#     return np.max(min_deltas)

# hausdorff_matrix = np.zeros((NN, NN))
# for ii in range(NN):
#     for jj in range(ii+1, NN):
#         hausdorff_matrix[ii, jj] = hausdorff_distance_between_images_nbdbased(all_embeddings[ii], all_embeddings[jj])

# # fill up the lower triangle
# hausdorff_matrix = np.triu(hausdorff_matrix) + np.triu(hausdorff_matrix, 1).T

# print("Hausdorff distances computed")


""" Compute MDS embeddings from the Hausdorff distance matrix """

# MDS_DIM = 8         # target embedding dimension
# MDS_EIG_TOL = 1e-8  # eigenvalue threshold

# def compute_classical_mds_embedding(D, dim=MDS_DIM, eig_tol=MDS_EIG_TOL):
#     """
#     Classical MDS embedding from a distance matrix.

#     Args:
#         D (np.ndarray): NxN symmetric distance matrix
#         dim (int): target embedding dimension
#         eig_tol (float): eigenvalue threshold

#     Returns:
#         X (np.ndarray): Nxk embedding (k <= dim)
#         eigvals (np.ndarray): eigenvalues used
#     """
#     D = np.asarray(D)
#     N = D.shape[0]

#     J = np.eye(N) - np.ones((N, N)) / N
#     B = -0.5 * J @ (D ** 2) @ J

#     eigvals, eigvecs = np.linalg.eigh(B)

#     # Sort descending
#     idx = np.argsort(eigvals)[::-1]
#     eigvals = eigvals[idx]
#     eigvecs = eigvecs[:, idx]

#     # Keep strictly positive eigenvalues
#     pos = eigvals > eig_tol
#     eigvals = eigvals[pos]
#     eigvecs = eigvecs[:, pos]

#     # Truncate to target dim
#     k = min(dim, eigvals.shape[0])
#     eigvals = eigvals[:k]
#     eigvecs = eigvecs[:, :k]

#     X = eigvecs @ np.diag(np.sqrt(eigvals))
#     return X, eigvals

# mds_embedding, _ = compute_classical_mds_embedding(hausdorff_matrix)

# print("MDS embeddings computed")


# """ Compute ZCore scores from the MDS embeddings """

# ctx = {
#     "dataset": dataset,
#     "view": dataset_slice,
#     "params": {
#         "embeddings": "embeddings_hausdorff_nbd_mds_8",
#         "zcore_score_field": "zcore_score_hausdorff_nbd_mds_8",
#     },
# }
# foo.execute_operator("@51labs/zero-shot-coreset-selection/compute_zcore_score", ctx)

# print("ZCore scores computed")


""" Visualize """

session = fo.launch_app(dataset_slice)
session.wait()


"""

Next steps:

1. Sample
2. Exemplar Assignment
3. Annotation Propagation
4. Evaluation

Given a constant annotation_propagation module, and a fixed selection budget γ, compare:

- random selection
- every (1/γ)th sample
- ZCore selection with clip-derived embeddings
- ZCore selection with hausdorff-mds-derived embeddings

"""