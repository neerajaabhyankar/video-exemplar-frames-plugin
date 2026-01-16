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
    from annoprop import propagate_annotations


class ExtractExemplarFrames(foo.Operator):
    @property
    def config(self):
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

        # # brain key
        # inputs.str(
        #     "exemplar_run_key",
        #     label="Brain Key for Exemplar Frame Extraction",
        #     default="extract_exemplar_frames",
        #     required=True,
        # )

        return types.Property(inputs)
    
    def execute(self, ctx):
        ctx.dataset.persistent = True  # Required for custom runs
        view = ctx.target_view()
        max_fraction_exemplars = ctx.params.get("max_fraction_exemplars", 0.1)
        exemplar_frame_field = ctx.params.get("exemplar_frame_field", "exemplar")
        # exemplar_run_key = ctx.params.get("exemplar_run_key", "extract_exemplar_frames")
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
            view,
            max_fraction_exemplars=max_fraction_exemplars,
            exemplar_frame_field=exemplar_frame_field,
            method=method,
        )
        
        # # Store as a custom run
        # config = ctx.dataset.init_run()
        # config.max_fraction_exemplars = max_fraction_exemplars
        # config.exemplar_frame_field = exemplar_frame_field
        # ctx.dataset.register_run(exemplar_run_key, config, overwrite=True, cleanup=False)
        # # TODO(neeraja): run cleanup should involve deleting the exemplar_frame_field
        # results = ctx.dataset.init_run_results(exemplar_run_key)
        # ctx.dataset.save_run_results(exemplar_run_key, results, overwrite=True)
        
        return {
            "message": f"Exemplar frames extracted and stored in field '{exemplar_frame_field}'",
        }

class PropagateAnnotationsFromExemplars(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="propagate_annotations_from_exemplars",
            label="Propagate Annotations From Exemplars Operator",
            description="Propagate annotations from exemplar frames to all the frames",
            icon="/assets/keyframes_anno.svg",
            dynamic=True,
        )

    def execute(self, ctx):
        # run_key = ctx.params.get("exemplar_run_key", "extract_exemplar_frames")
        # # Load the exemplar_assignments from the custom run
        # try:
        #     run_info = ctx.dataset.get_run_info(run_key)
        #     stored_exemplar_frame_field = run_info.config.exemplar_frame_field
        # except Exception as e:
        #     raise ValueError(f"Could not load exemplar run '{run_key}': {e}. "
        #                    f"Make sure you have run 'extract_exemplar_frames' first.")

        exemplar_frame_field = ctx.params.get("exemplar_frame_field", "exemplar")

        input_annotation_field = ctx.params.get("input_annotation_field", "human_labels")
        output_annotation_field = ctx.params.get("output_annotation_field", "human_labels_propagated")

        if output_annotation_field == input_annotation_field:
            raise ValueError("Output annotation field cannot be the same as the input annotation field")
        
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

    def resolve_input(self, ctx):
        inputs = types.Object()

        # # Check if there are any extract_exemplar_frames runs
        # available_runs = ctx.dataset.list_runs()
        # available_runs = [
        #     rr
        #     for rr in available_runs
        #     if hasattr(ctx.dataset.get_run_info(rr).config, "exemplar_frame_field")
        # ]

        # if len(available_runs) > 0:
        #     run_dropdown = types.Dropdown()
        #     for run_key in available_runs:
        #         try:
        #             run_info = ctx.dataset.get_run_info(run_key)
        #             # Create a label from the run info
        #             timestamp = run_info.get("timestamp", "unknown")
        #             exemplar_field = run_info.get("config", {}).get("exemplar_frame_field", "unknown")
        #             # Format timestamp for display (just date and time, no microseconds)
        #             if timestamp != "unknown":
        #                 timestamp = timestamp.split(".")[0].replace("T", " ")
        #             label = f"{run_key} ({exemplar_field}, {timestamp})"
        #             run_dropdown.add_choice(run_key, label=label)
        #         except Exception:
        #             # Fallback if we can't get run info
        #             run_dropdown.add_choice(run_key, label=run_key)
            
        #     inputs.enum(
        #         "exemplar_run_key",
        #         run_dropdown.values(),
        #         default=available_runs[0] if available_runs else None,
        #         label="Exemplar Extraction Run Key",
        #         view=run_dropdown,
        #         required=True,
        #     )
        # else:
        #     raise ValueError("No exemplar extraction runs found. Please run 'extract_exemplar_frames' first.")

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
