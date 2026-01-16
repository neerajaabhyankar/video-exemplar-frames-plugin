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
from utils import evaluate
from annoprop_algos import propagate_detections_with_siamese


""" Load dataset """

dataset = fo.load_dataset("basketball_frames")
dataset_slice = dataset.load_saved_view("spinning_part1")
# dataset_slice = dataset.load_saved_view("side_top_layup")
# dataset_slice = dataset.load_saved_view("underbasket_reverse_layup")
print(len(dataset_slice))


"""
Assume the following are done (see test_zcore_siamfc_e2e.py for details):
- Compute SIAMFC embeddings
- Compute Hausdorff distances
- Compute MDS embedding
"""

"""
Now,
1. Choose a random exemplar
2. Assign it to all the samples in the dataset
3. Compute the accuracy of the propagated annotations
4. Plot accuracy vs distance in embedding space
"""

embedding_field_name = "embeddings_hausdorff_nbd_mds_8"


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
        all_samples._dataset.add_sample_field(exemplar_indicator_field, fo.DictField)

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
            is_exemplar = True
        else:
            is_exemplar = False
        sample[exemplar_indicator_field] = {
            "is_exemplar": is_exemplar,
            "exemplar_assignment": [nnbr_id] if not is_exemplar else []
        }
        sample.save()
    
    all_samples.save()
    return all_samples


chosen_exemplar = dataset_slice.take(1, seed=42)
if "exemplar_single" in dataset_slice._dataset.get_field_schema():
    dataset_slice._dataset.delete_sample_field("exemplar_single")
dataset_slice = assign_to_nearest_neighbor_exemplar(
    dataset_slice,
    chosen_exemplar,
    embedding_field_name,
    "exemplar_single",
)


def propagate_annotations_dict(
    view,
    exemplar_frame_field: str,
    input_annotation_field: str,
    output_annotation_field: str,
    evaluate_propagation = True,
) -> dict:
    """
    Propagate annotations from exemplar frames to all the frames.
    Args:
        view: The view to propagate annotations from
        exemplar_frame_field: The field name of the exemplar frame
                              TODO(neeraja): Explore whether we can remove this field
        annotation_field: The field name of the annotation to copy from the exemplar frame
        output_annotation_field: The field name of the annotation to save to the target frame
        exemplar_assignments: {sample_id: [exemplar_frame_ids]} for each sample in the view
        evaluate_propagation: Whether to evaluate the propagation against
                              the input annotation field present in the propagation targets.
    """
    scores = {}

    for sample in view:
        if sample[exemplar_frame_field]["is_exemplar"]:
            sample[output_annotation_field] = sample[input_annotation_field]
        elif len(sample[exemplar_frame_field]["exemplar_assignment"]) > 0:
            exemplar_frame_ids = sample[exemplar_frame_field]["exemplar_assignment"]

            # TODO(neeraja): handle multiple exemplar frames for the same sample
            exemplar_sample = view[exemplar_frame_ids[0]]
            exemplar_frame = cv2.imread(exemplar_sample.filepath)
            exemplar_detections = exemplar_sample[input_annotation_field]

            sample_frame = cv2.imread(sample.filepath)
            propagated_detections = propagate_detections_with_siamese(exemplar_frame, sample_frame, exemplar_detections)
            sample[output_annotation_field] = propagated_detections
            sample.save()

            # If the sample already has an input annotation field, evaluate against it
            if evaluate_propagation and sample[input_annotation_field]:
                original_detections = sample[input_annotation_field]
                # TODO(neeraja): decouple the matching and the evaluation
                sample_score = evaluate(original_detections, propagated_detections)
                scores[sample.id] = sample_score
    
    return scores


score_dict = propagate_annotations_dict(
    dataset_slice,
    exemplar_frame_field="exemplar_single",
    input_annotation_field="ha_test_1",
    output_annotation_field="ha_test_1_propagated",
)
sorted_scores = [score_dict[k] for k in sorted(score_dict.keys())]


exemplar_embedding = chosen_exemplar.values(embedding_field_name)[0]
siamfc_distances = [
    np.linalg.norm(sample[embedding_field_name] - exemplar_embedding)
    for sample in dataset_slice
    if sample.id in score_dict
]

exemplar_embedding = chosen_exemplar.values("embeddings_clip")[0]
clip_distances = [
    np.linalg.norm(sample["embeddings_clip"] - exemplar_embedding)
    for sample in dataset_slice
    if sample.id in score_dict
]

exemplar_embedding = chosen_exemplar.values("embeddings_resnet18")[0]
resnet18_distances = [
    np.linalg.norm(sample["embeddings_resnet18"] - exemplar_embedding)
    for sample in dataset_slice
    if sample.id in score_dict
]

# plt.scatter(sorted_scores, siamfc_distances, c="seagreen")  
# plt.scatter(sorted_scores, resnet18_distances, c="lightcoral")
# plt.scatter(sorted_scores, clip_distances, c="orchid")
# plt.xlabel("Propagation IoU Score")
# plt.ylabel("Distance to Exemplar in SiamFC Embedding Space")
# plt.title("Distances in Propagatability Space v/s Score")
# plt.show()


# Compute correlation coefficients for the three distances
# Since lower distance should correlate with higher score, we use negative distances
from scipy.stats import spearmanr, pearsonr

# Convert to numpy arrays
scores_array = np.array(sorted_scores)
siamfc_array = np.array(siamfc_distances)
resnet18_array = np.array(resnet18_distances)
clip_array = np.array(clip_distances)

# Spearman correlation (rank-based, measures monotonic relationship)
# Use negative distances since lower distance should mean higher score
spearman_siamfc, p_siamfc = spearmanr(scores_array, -siamfc_array)
spearman_resnet18, p_resnet18 = spearmanr(scores_array, -resnet18_array)
spearman_clip, p_clip = spearmanr(scores_array, -clip_array)

# Pearson correlation (linear relationship)
pearson_siamfc, p_pearson_siamfc = pearsonr(scores_array, -siamfc_array)
pearson_resnet18, p_pearson_resnet18 = pearsonr(scores_array, -resnet18_array)
pearson_clip, p_pearson_clip = pearsonr(scores_array, -clip_array)

print("Correlation with Propagation Scores (higher = better correlation):")
print(f"\nSpearman (rank correlation):")
print(f"  SiamFC:   {spearman_siamfc:.3f} (p={p_siamfc:.4f})")
print(f"  ResNet18: {spearman_resnet18:.3f} (p={p_resnet18:.4f})")
print(f"  CLIP:     {spearman_clip:.3f} (p={p_clip:.4f})")
print(f"\nPearson (linear correlation):")
print(f"  SiamFC:   {pearson_siamfc:.3f} (p={p_pearson_siamfc:.4f})")
print(f"  ResNet18: {pearson_resnet18:.3f} (p={p_pearson_resnet18:.4f})")
print(f"  CLIP:     {pearson_clip:.3f} (p={p_pearson_clip:.4f})")

# Visualize correlations
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(resnet18_array, scores_array, alpha=0.6, c='lightcoral')
plt.xlabel('ResNet18 Distance')
plt.ylabel('Propagation Score')
plt.title(f'ResNet18 (Spearman: {spearman_resnet18:.3f})')
plt.grid(True, alpha=0.7)

plt.subplot(1, 3, 2)
plt.scatter(clip_array, scores_array, alpha=0.6, c='orchid')
plt.xlabel('CLIP Distance')
plt.ylabel('Propagation Score')
plt.title(f'CLIP (Spearman: {spearman_clip:.3f})')
plt.grid(True, alpha=0.7)

plt.subplot(1, 3, 3)
plt.scatter(siamfc_array, scores_array, alpha=0.6, c='seagreen')
plt.xlabel('SiamFC Distance')
plt.ylabel('Propagation Score')
plt.title(f'SiamFC (Spearman: {spearman_siamfc:.3f})')
plt.grid(True, alpha=0.7)

plt.tight_layout()
plt.show()