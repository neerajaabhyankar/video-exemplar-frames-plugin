"""Video Exemplar Frames plugin.

Extract exemplar frames from a video dataset and propagate annotations.

| Copyright 2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`
"""

import os

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone.core.utils import add_sys_path

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from exemplars import extract_exemplar_frames
    from annoprop import propagate_annotations


class ExtractExemplarFrames(foo.Operator):
    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="extract_exemplar_frames", 
            label="Extract Exemplar Frames Operator",
            description="Extract exemplar frames from a video dataset",
            icon="/assets/keyframes.svg",
            dynamic=False,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # exemplar extraction method
        # TODO(neeraja): this is hacky right now, will be made cleaner
        method_choices = [
            "random",
            "uniform",
            "hdbscan:embeddings_resnet18",
            "hdbscan:embeddings_clip",
            "hdbscan:embeddings_hausdorff_nbd_mds_8",
            "zcore:embeddings_resnet18",
            "zcore:embeddings_clip",
            "zcore:embeddings_hausdorff_nbd_mds_8",
        ]
        method_dropdown = types.Dropdown()
        for choice in method_choices:
            method_dropdown.add_choice(choice, label=choice)
        
        inputs.enum(
            "method",
            method_dropdown.values(),
            default="random",
            label="Exemplar Extraction Method",
            view=method_dropdown,
            required=True,
        )

        # max fraction of exemplars to extract
        inputs.float(
            "max_fraction_exemplars",
            label="Max Fraction of Exemplars",
            default=0.1,
            min=0.0,
            max=1.0,
            view=types.SliderView(min=0.0, max=1.0, step=0.05),
            required=True,
        )

        # exemplar frame field
        inputs.str(
            "exemplar_frame_field",
            label="Exemplar Frame Information Field",
            default="exemplar",
            required=True,
        )

        return types.Property(inputs)
    
    def execute(self, ctx) -> dict:
        ctx.dataset.persistent = True  # Required for custom runs
        max_fraction_exemplars = ctx.params.get("max_fraction_exemplars", 0.1)
        exemplar_frame_field = ctx.params.get("exemplar_frame_field", "exemplar")
        method = ctx.params.get("method", "random")

        # Check if field exists and validate/convert its type
        dataset = ctx.dataset
        if exemplar_frame_field in dataset.get_field_schema():
            field_type_name = type(dataset.get_field_schema()[exemplar_frame_field]).__name__
            if field_type_name != "EmbeddedDocumentField":
                dataset.delete_sample_field(exemplar_frame_field)
        
        # Ensure the exemplar field exists and declare nested fields for proper schema support
        # Use DynamicEmbeddedDocument to allow dynamic fields
        if exemplar_frame_field not in dataset.get_field_schema():
            dataset.add_sample_field(
                exemplar_frame_field, 
                fo.EmbeddedDocumentField,
                embedded_doc_type=fo.DynamicEmbeddedDocument
            )
            dataset.add_sample_field(f"{exemplar_frame_field}.is_exemplar", fo.BooleanField)
            dataset.add_sample_field(
                f"{exemplar_frame_field}.exemplar_assignment", 
                fo.ListField, subfield=fo.ObjectIdField
            )

        extract_exemplar_frames(
            ctx.target_view(),
            max_fraction_exemplars=max_fraction_exemplars,
            exemplar_frame_field=exemplar_frame_field,
            method=method,
        )
        
        return {
            "message": f"Exemplar frames extracted and stored in field '{exemplar_frame_field}'",
        }

class PropagateAnnotationsFromExemplars(foo.Operator):
    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="propagate_annotations_from_exemplars",
            label="Propagate Annotations From Exemplars Operator",
            description="Propagate annotations from exemplar frames to all the frames",
            icon="/assets/keyframes_anno.svg",
            dynamic=True,
        )

    def execute(self, ctx):
        exemplar_frame_field = ctx.params.get("exemplar_frame_field", "exemplar")

        input_annotation_field = ctx.params.get("input_annotation_field", "human_labels")
        output_annotation_field = ctx.params.get("output_annotation_field", "human_labels_propagated")

        if output_annotation_field == input_annotation_field:
            raise ValueError(
                f"Output annotation field '{output_annotation_field}' cannot be the same as "
                f"the input annotation field '{input_annotation_field}'. "
                f"Please choose a different output field name to avoid overwriting the source annotations."
            )
        
        # TODO(neeraja): allow for (re?)computing exemplar assignments
        # at this stage, given the annotations!

        propagation_score = propagate_annotations(
            view=ctx.target_view(),
            exemplar_frame_field=exemplar_frame_field,
            input_annotation_field=input_annotation_field,
            output_annotation_field=output_annotation_field,
        )
        return {
            "propagation_score": propagation_score,
            "message": f"Annotations propagated from {input_annotation_field} to {output_annotation_field}",
        }

    def resolve_input(self, ctx) -> types.Property:
        inputs = types.Object()
        
        inputs.str(
            "exemplar_frame_field",
            label="Exemplar Frame Information Field",
            default="exemplar",
            required=True,
        )

        inputs.str(
            "input_annotation_field",
            label="Annotation Field to Propagate from",
            default="human_labels",
            required=True,
        )

        inputs.str(
            "output_annotation_field",
            label="Annotation Field to Propagate to",
            default="human_labels_propagated",
            required=True,
        )

        return types.Property(inputs)


def register(p):
    p.register(ExtractExemplarFrames)
    p.register(PropagateAnnotationsFromExemplars)
