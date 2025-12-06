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

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from exemplars import extract_exemplar_frames

def _input_control_flow(ctx):
    inputs = types.Object()

    # max fraction of exemplars to extract
    inputs.float(
        "max_fraction_exemplars",
        label="Max Fraction of Exemplars",
        default=0.1,
        min=0.0,
        max=1.0,
        view=types.SliderView(min=0.0, max=1.0, step=0.05),
    )

    # exemplar frame field
    inputs.str(
        "exemplar_frame_field",
        label="Exemplar Frame Field",
        default="exemplar",
        view=types.TextView(placeholder="e.g. 'exemplar1'"),
    )

    return types.Property(inputs)

class ExtractExemplarFrames(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="extract_exemplar_frames", 
            label="Extract Exemplar Frames Operator",
            description="Extract exemplar frames from a video dataset",
            icon="/assets/keyframes.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        return _input_control_flow(ctx)

    def execute(self, ctx):
        # ctx.dataset.persistent = True
        view = ctx.target_view()
        max_fraction_exemplars = ctx.params.get("max_fraction_exemplars")
        exemplar_frame_field = ctx.params.get("exemplar_frame_field")
        extract_exemplar_frames(
            view,
            max_fraction_exemplars=max_fraction_exemplars,
            exemplar_frame_field=exemplar_frame_field,
        )
        return {}


def register(p):
    p.register(ExtractExemplarFrames)
