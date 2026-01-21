import logging
import random
from typing import Union
import numpy as np

import fiftyone as fo

logger = logging.getLogger(__name__)


def extract_exemplar_frames(
    view: Union[fo.core.dataset.Dataset, fo.core.view.DatasetView],
    max_fraction_exemplars: float = 1.0,
    exemplar_frame_field: str = "exemplar",
    method: str = "random",
) -> None:
    logger.info(f"Extracting exemplar frames from {view.head(1)}")
    logger.info(f"Max fraction of exemplars: {max_fraction_exemplars}")
    logger.info(f"Exemplar frame field: {exemplar_frame_field}")
    logger.info(f"View: {view}")

    # # Case 1: Collection of images

    # get the number of frames in the view
    num_frames = view.count()
    logger.info(f"Number of samples in view: {num_frames}")
    
    if num_frames == 0:
        raise ValueError(
            "The view has no samples. Please select samples or adjust your view filters "
            "before extracting exemplar frames."
        )

    # get the number of frames to extract
    num_frames_to_extract = int(num_frames * max_fraction_exemplars)
    logger.info(f"Number of frames to extract: {num_frames_to_extract}")
    
    if num_frames_to_extract == 0:
        raise ValueError(
            f"With max_fraction_exemplars={max_fraction_exemplars} and {num_frames} samples, "
            f"no exemplar frames would be extracted. Please increase max_fraction_exemplars "
            f"(minimum: {1.0 / num_frames if num_frames > 0 else 0.01:.4f})."
        )

    if method == "random":
        # first frame is an exemplar
        curr_exemplar_id = view.first().id
        for sample in view:
            if random.random() < max_fraction_exemplars:
                curr_exemplar_id = sample.id
                is_exemplar = True
            else:
                is_exemplar = False
            sample[exemplar_frame_field] = fo.DynamicEmbeddedDocument(
                is_exemplar=is_exemplar,
                exemplar_assignment=[curr_exemplar_id] if not is_exemplar else []
            )
            sample.save()
    
    elif method == "uniform":
        # every (1/γ)th sample is an exemplar
        # first frame is an exemplar
        curr_exemplar_id = view.first().id
        for ii, sample in enumerate(view):
            if ii % int(num_frames / num_frames_to_extract) == 0:
                curr_exemplar_id = sample.id
                is_exemplar = True
            else:
                is_exemplar = False
            sample[exemplar_frame_field] = fo.DynamicEmbeddedDocument(
                is_exemplar=is_exemplar,
                exemplar_assignment=[curr_exemplar_id] if not is_exemplar else []
            )
            sample.save()
    
    else:
        clustering_method, embedding_field = method.split(":")
        
        if clustering_method not in ["hdbscan", "zcore"]:
            raise ValueError(
                f"Unknown clustering method '{clustering_method}'. "
                f"Supported methods: 'hdbscan', 'zcore'. "
                f"Format: '{clustering_method}:embedding_field'."
            )

        if not hasattr(view.first(), embedding_field):
            available_fields = list(view.first().get_field_schema().keys())
            raise ValueError(
                f"Embedding field '{embedding_field}' not found in the dataset. "
                f"Available fields: {', '.join(available_fields[:10])}{'...' if len(available_fields) > 10 else ''}. "
                f"Please compute embeddings first using a brain run or embedding operator (e.g., "
                f"'@voxel51/brain/compute_visualization' or '@voxel51/zoo/compute_embeddings')."
            )
        
        if clustering_method == "hdbscan":
            expected_num_clusters = int(num_frames * max_fraction_exemplars)
            expected_num_clusters = expected_num_clusters / 2  # allow for some noisy samples
            min_cluster_size = int(num_frames / expected_num_clusters)

            from sklearn.cluster import HDBSCAN
            np.random.seed(42)
            hdbscan = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=3,
                cluster_selection_epsilon=0.5,
                store_centers="medoid",
            )
            hdbscan.fit(view.values(embedding_field))
            cluster_labels = hdbscan.labels_  # these are indices
            exemplar_vectors = hdbscan.medoids_  # these are vectors


            # find the sample IDs corresponding to the chosen exemplars
            # distance_matrix has rows = len(medoids) and columns = len(samples)
            distance_matrix = np.array([np.linalg.norm(view.values(embedding_field) - exemplar_vector, axis=1) for exemplar_vector in exemplar_vectors])
            sample_ids = view.values("id")
            cluster_label_to_exemplar_id = {
                ii: sample_ids[np.argmin(distance_matrix[ii])]
                for ii in range(len(exemplar_vectors))
            }

            # populate assignments
            for ii, sample in enumerate(view):
                if cluster_labels[ii] == -1:
                    is_exemplar = True
                    assignment = []
                else:
                    if sample.id in cluster_label_to_exemplar_id.values():
                        is_exemplar = True
                    else:
                        is_exemplar = False
                sample[exemplar_frame_field] = fo.DynamicEmbeddedDocument(
                    is_exemplar=is_exemplar,
                    exemplar_assignment=[cluster_label_to_exemplar_id[cluster_labels[ii]]] if not is_exemplar else []
                )
                sample.save()

        elif clustering_method == "zcore":
            import fiftyone.operators as foo
            zcore_score_field_name = f"zcore_score_{embedding_field.replace('embeddings_', '')}"
            ctx = {
                "dataset": view._dataset,
                "view": view,
                "params": {
                    "embeddings": embedding_field,
                    "zcore_score_field": zcore_score_field_name,
                },
            }
            foo.execute_operator("@51labs/zero-shot-coreset-selection/compute_zcore_score", ctx)

            exemplar_samples = view.sort_by(
                zcore_score_field_name, reverse=True
            )[:num_frames_to_extract]
            exemplar_ids = exemplar_samples.values("id")
            exemplar_embeddings = exemplar_samples.values(embedding_field)


            # populate assignments
            for sample in view:
                nnbr_index = np.argmin(
                    np.linalg.norm(
                        exemplar_embeddings - sample[embedding_field], axis=1
                    )
                )
                nnbr_id = exemplar_ids[nnbr_index]
                # Assign the sample to the nearest neighbor
                is_exemplar = (nnbr_id == sample.id)
                sample[exemplar_frame_field] = fo.DynamicEmbeddedDocument(
                    is_exemplar=is_exemplar,
                    exemplar_assignment=[nnbr_id] if not is_exemplar else []
                )
                sample.save()
    
    # # Case 2: Collection of videos

    # for sample in view:
    #     for frame in sample.frames:
    #         if random.random() < max_fraction_exemplars:
    #             frame[exemplar_frame_field] = True
    #         else:
    #             frame[exemplar_frame_field] = False

    return