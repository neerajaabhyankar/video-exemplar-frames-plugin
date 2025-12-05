"""Image Redaction plugin.

| Copyright 2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`
"""

import os
import re

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.utils.transformers as fout
from fiftyone.core.utils import add_sys_path

# with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
#     from exemplars import extract_exemplar_frames

def _input_control_flow(ctx):
    return ctx.dataset.view()

class ExtractExemplarFrames(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="extract_exemplar_frames", 
            label="Extract Exemplar Frames Operator",
            description="Extract exemplar frames from a video dataset",
            # icon="/assets/video.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        return _input_control_flow(ctx)

    def execute(self, ctx):
        ctx.dataset.persistent = True
        view = ctx.target_view()
        
        # extract_exemplar_frames(
        #     view,
        # )
        return {}


def register(p):
    p.register(ExtractExemplarFrames)
