import logging
import random
from typing import Union

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

    # get the number of frames in the view
    num_frames = view.count()
    logger.info(f"Number of frames in view: {num_frames}")
    # ensure this is a video dataset
    breakpoint()
    if not view.has_field("frames"):
        raise ValueError("View is not a video dataset")

    # get the number of frames to extract
    num_frames_to_extract = int(num_frames * max_fraction_exemplars)
    logger.info(f"Number of frames to extract: {num_frames_to_extract}")

    # TODO(neeraja): replace this random sampling

    for sample in view:
        for frame in sample.frames:
            if random.random() < max_fraction_exemplars:
                frame[exemplar_frame_field] = True
            else:
                frame[exemplar_frame_field] = False

    return None