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
) -> None:
    logger.info(f"Extracting exemplar frames from {view.head(1)}")
    logger.info(f"Max fraction of exemplars: {max_fraction_exemplars}")
    logger.info(f"Exemplar frame field: {exemplar_frame_field}")
    logger.info(f"View: {view}")

    # # Case 1: Collection of images

    # get the number of frames in the view
    num_frames = view.count()
    logger.info(f"Number of samples in view: {num_frames}")

    # get the number of frames to extract
    num_frames_to_extract = int(num_frames * max_fraction_exemplars)
    logger.info(f"Number of frames to extract: {num_frames_to_extract}")

    exemplar_assignments = {}  # {sample_id: [exemplar_frame_ids]}


    # TODO(neeraja): use embeddings to extract exemplars

    if hasattr(view.first(), "embeddings"):
        # use embeddings to extract exemplars

        from sklearn.cluster import HDBSCAN

        np.random.seed(42)

        hdbscan = HDBSCAN(
            min_cluster_size=4,
            min_samples=3,
            cluster_selection_epsilon=0.5,
            store_centers="medoid",
        )
        hdbscan.fit(view.values("embeddings"))
        cluster_labels = hdbscan.labels_  # these are indices
        cluster_medoids = hdbscan.medoids_  # these are vectors

        # find the sample IDs corresponding to the cluster medoids
        # distance_matrix has rows = len(medoids) and columns = len(samples)
        distance_matrix = np.array([np.linalg.norm(view.values("embeddings") - medoid, axis=1) for medoid in cluster_medoids])
        sample_ids = view.values("id")
        exemplar_sample_ids = {
            ii: sample_ids[np.argmin(distance_matrix[ii])]
            for ii in range(len(cluster_medoids))
        }

        for ii, sample in enumerate(view):
            if cluster_labels[ii] == -1:
                sample[exemplar_frame_field] = True
                exemplar_assignments[sample.id] = [sample.id]
            else:
                if sample.id in exemplar_sample_ids.values():
                    sample[exemplar_frame_field] = True
                else:
                    sample[exemplar_frame_field] = False
                exemplar_assignments[sample.id] = [exemplar_sample_ids[cluster_labels[ii]]]
            sample.save()

    else:
        # TODO(neeraja): fallback to random sampling

        # first frame is an exemplar
        curr_exemplar_id = view.first().id
        for sample in view:
            if random.random() < max_fraction_exemplars:
                curr_exemplar_id = sample.id
                sample[exemplar_frame_field] = True
            else:
                sample[exemplar_frame_field] = False
            sample.save()
            exemplar_assignments[sample.id] = [curr_exemplar_id]

    # # Case 2: Collection of videos

    # for sample in view:
    #     for frame in sample.frames:
    #         if random.random() < max_fraction_exemplars:
    #             frame[exemplar_frame_field] = True
    #         else:
    #             frame[exemplar_frame_field] = False

    return exemplar_assignments