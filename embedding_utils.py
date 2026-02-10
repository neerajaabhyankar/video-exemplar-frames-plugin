from typing import Tuple, Union, Optional
from tqdm import tqdm
import numpy as np
import cv2
import logging

import fiftyone as fo

from  suc_utils import normalized_bbox_to_pixel_coords

fo.config.database_validation = False
logger = logging.getLogger(__name__)


BACKBONE_INPUT_SIZE = 255

def preprocess_for_siamfc(img: np.ndarray, out_size: int = BACKBONE_INPUT_SIZE, device=None) -> np.ndarray:
    """
    img: np.ndarray (3, H, W).
    Returns: torch.Tensor of shape (1, 3, out_size, out_size), float32.
    """
    # Ensure HWC for OpenCV resize
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {img.shape}")

    # Resize to out_size x out_size (warps aspect ratio, like SiamFC)
    img_resized = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    return img_resized


def compute_backbone_embeddings_siamfc(
    frames: Union[fo.core.collections.SampleCollection, list[np.ndarray]],
    spatial_embedding_field_name: str = "embeddings_siamfc",
) -> np.ndarray:
    """
    Compute the backbone embeddings for the given frames using SiamFC.
    Args:
        frames: A fiftyone SampleCollection or list of numpy arrays containing the frames to compute the embeddings for.
        spatial_embedding_field_name: The field in which to save the spatial embeddings of the image
    Returns:
        None
        Populates the `spatial_embedding_field_name` field with the embeddings of the frames.
        Each has shape (DD, HH, WW), where DD is the embedding dimension, HH, WW are the spatial dimensions.
    """
    import torch
    from labelprop_methods.siamese import PropagatorSiamFC

    propagator = PropagatorSiamFC()
    tracker = propagator.setup()
    _ = tracker.net.eval()
    
    def compute_embedding(img: np.ndarray) -> np.ndarray:
        # Convert to torch tensor and process through backbone (similar to siamfc.py:159-166)
        # but without cropping - send entire image
        img = preprocess_for_siamfc(img)
        x = torch.from_numpy(img).to(
            tracker.device).permute(2, 0, 1).unsqueeze(0).float()
        embedding = tracker.net.backbone(x)
        embedding = embedding.detach().cpu().numpy().squeeze(0)
        return embedding
    
    logger.info(f"Computing backbone embeddings for {len(frames)} frames")

    if isinstance(frames, fo.core.collections.SampleCollection):
        def populate_embedding(sample: fo.core.sample.Sample):
            img = cv2.imread(sample.filepath)
            embedding = compute_embedding(img)
            sample[spatial_embedding_field_name] = embedding
            return sample
        
        # _ = list(frames.map_samples(populate_embedding, save=True))
        # TODO(neeraja): look into why mapping fails
        for sample in frames:
            populate_embedding(sample)
            sample.save()
        return None
    else:
        embeddings = [compute_embedding(frame) for frame in frames]
        return embeddings


def hausdorff_distance_between_images_nbdbased(img_emb_1, img_emb_2):
    """
    Compute the Hausdorff distance between two images based on their embeddings.
    Args:
        img_emb_1: First image embedding numpy array of shape (DD, HH, WW)
        img_emb_2: Second image embedding numpy array of shape (DD, HH, WW)
    Returns:
        float: A modified Hausdorff distance between the two images.
        Instead of computing the min deltas over the entire image,
        we compute the min deltas over the corresponding neighborhood in the other image,
        for each patch in one image.
    """
    DD, HH, WW = img_emb_1.shape
    PATCH_NBD = max(HH, WW) // 10
    min_deltas = []
    for hh in range(HH):
        for ww in range(WW):
            patch2 = img_emb_2[:, hh, ww]
            img1_neighborhood = img_emb_1[
                :,
                max(0, hh-PATCH_NBD):min(img_emb_1.shape[1], hh+PATCH_NBD),
                max(0, ww-PATCH_NBD):min(img_emb_1.shape[2], ww+PATCH_NBD),
            ]
            patch_deltas = np.linalg.norm(img1_neighborhood - patch2[:, np.newaxis, np.newaxis], axis=0)
            min_deltas.append(np.min(patch_deltas))
    return np.max(min_deltas)


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
    """
    logger.info(f"Computing classical MDS embedding for {D.shape[0]} frames")

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
    return X


def compute_hausdorff_distance_matrix(
    frames: fo.core.collections.SampleCollection,
    spatial_embedding_field_name: str = "embeddings_siamfc",
) -> np.ndarray:
    """
    Args:
        frames: A fiftyone SampleCollection
        spatial_embedding_field_name: The field containing spatial embeddings of the image, from a model backbone.
    Returns:
        A numpy array of shape (NN, NN) containing the Hausdorff distance matrix.
        where NN is the number of frames.
    """
    NN = len(frames)

    logger.info(f"Computing Hausdorff distance matrix for {NN} frames (quadratic complexity)")
    hausdorff_matrix = np.zeros((NN, NN))
    for ii, frame in tqdm(enumerate(frames)):
        for jj, other_frame in enumerate(frames):
            if ii >= jj:
                continue
            hausdorff_matrix[ii, jj] = hausdorff_distance_between_images_nbdbased(
                frame[spatial_embedding_field_name], other_frame[spatial_embedding_field_name]
            )

    # fill up the lower triangle
    hausdorff_matrix = np.triu(hausdorff_matrix) + np.triu(hausdorff_matrix, 1).T
    
    return hausdorff_matrix


def compute_hausdorff_mds_embedding(
    frames: fo.core.collections.SampleCollection,
    spatial_embedding_field_name: str = "embeddings_siamfc",
    mds_embedding_field_name: str = "embeddings_hausdorff_mds_siamfc_8",
) -> np.ndarray:
    """
    Args:
        frames: A fiftyone SampleCollection
        spatial_embedding_field_name: The field containing spatial embeddings of the image, from a model backbone.
        mds_embedding_field_name: The field name of the output MDS embeddings
    Returns:
        None
    Populates the `field_name` of the frames with the output of the
    Hausdorff embedding computed via MDS with the SiamFC backbone.
    Dimension MDS_DIM.
    """
    hausdorff_matrix = compute_hausdorff_distance_matrix(frames, spatial_embedding_field_name)
    mds_embedding = compute_classical_mds_embedding(hausdorff_matrix)

    logger.info(f"Populating {mds_embedding_field_name} with embeddings of shape {mds_embedding.shape}")
    frames.set_values(mds_embedding_field_name, mds_embedding)
    frames.save()


def propagatability_pre_label(source_frame: np.ndarray, target_frame: np.ndarray):
    """
    Args:
        source_frame: The source frame array
        target_frame: The target frame array
    Returns:
        The propagatability score
    """
    source_embedding, target_embedding = compute_backbone_embeddings_siamfc([source_frame, target_frame])
    hausdorff_distance = hausdorff_distance_between_images_nbdbased(
        source_embedding, target_embedding
    )
    return 1.0 / hausdorff_distance


def propagatability_post_label(source_frame: np.ndarray, target_frame: np.ndarray, source_detections: fo.Detections):
    """
    Args:
        source_frame: The source frame array
        target_frame: The target frame array
        source_detections: The source detections
    Returns:
        The propagatability score
    """
    source_embedding, target_embedding = compute_backbone_embeddings_siamfc([source_frame, target_frame])
    DD, HH, WW = source_embedding.shape
    # max_{all detections}[min_{all target patches}[distance(detection patch, target patch)]]
    # TODO(neeraja): test
    max_distance = 0.0
    for detection in source_detections.detections:
        detection_patch_corners = normalized_bbox_to_pixel_coords(detection.bounding_box, WW, HH)  # x1, y1, x2, y2
        detection_patch = source_embedding[
            :,
            detection_patch_corners[1]:detection_patch_corners[3],
            detection_patch_corners[0]:detection_patch_corners[2]
        ]

        patch_h = detection_patch.shape[1]
        patch_w = detection_patch.shape[2]
        num_y = HH - patch_h + 1
        num_x = WW - patch_w + 1
        target_emb = target_embedding
        s0, s1, s2 = target_emb.strides
        patches = np.lib.stride_tricks.as_strided(
            target_emb,
            shape=(DD, num_y, num_x, patch_h, patch_w),
            strides=(s0, s1, s2, s1, s2),
            writeable=False,
        )
        diffs = patches - detection_patch[:, None, None, :, :]
        diffs_flat = diffs.transpose(1, 2, 0, 3, 4).reshape(num_y, num_x, -1)
        dists = np.linalg.norm(diffs_flat, axis=-1)

        min_distance = dists.min()
        max_distance = max(max_distance, min_distance)
    
    return 1.0 / max_distance
