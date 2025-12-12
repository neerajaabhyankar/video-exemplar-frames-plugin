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
            label="Exemplar Frame Field (indicator field)",
            default="exemplar",
            required=True,
        )

        # brain key
        inputs.str(
            "exemplar_run_key",
            label="Brain Key for Exemplar Frame Extraction",
            default="extract_exemplar_frames",
            required=True,
        )

        return types.Property(inputs)
    
    def execute(self, ctx):
        ctx.dataset.persistent = True  # Required for custom runs
        view = ctx.target_view()
        max_fraction_exemplars = ctx.params.get("max_fraction_exemplars", 0.1)
        exemplar_frame_field = ctx.params.get("exemplar_frame_field", "exemplar")
        exemplar_run_key = ctx.params.get("exemplar_run_key", "extract_exemplar_frames")

        exemplar_assignments = extract_exemplar_frames(
            view,
            max_fraction_exemplars=max_fraction_exemplars,
            exemplar_frame_field=exemplar_frame_field,
        )
        
        # Store exemplar_assignments as a custom run
        config = ctx.dataset.init_run()
        config.max_fraction_exemplars = max_fraction_exemplars
        config.exemplar_frame_field = exemplar_frame_field
        ctx.dataset.register_run(exemplar_run_key, config, overwrite=True, cleanup=False)
        # TODO(neeraja): run cleanup should involve deleting the exemplar_frame_field
        
        results = ctx.dataset.init_run_results(exemplar_run_key)
        results.exemplar_assignments = exemplar_assignments
        ctx.dataset.save_run_results(exemplar_run_key, results, overwrite=True)
        
        return {
            "run_key": exemplar_run_key,
            "message": f"Exemplar frames extracted and stored in run '{exemplar_run_key}'",
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
        run_key = ctx.params.get("exemplar_run_key", "extract_exemplar_frames")
        
        # Load the exemplar_assignments from the custom run
        try:
            results = ctx.dataset.load_run_results(run_key)
            exemplar_assignments = results.exemplar_assignments
            run_info = ctx.dataset.get_run_info(run_key)
            stored_exemplar_frame_field = run_info.config.exemplar_frame_field
        except Exception as e:
            raise ValueError(f"Could not load exemplar run '{run_key}': {e}. "
                           f"Make sure you have run 'extract_exemplar_frames' first.")
        
        annotation_field = ctx.params.get("annotation_field", "human_labels")
        
        # Now exemplar_assignments is available for use in propagation logic
        # TODO: implement propagation logic using exemplar_assignments
        # exemplar_assignments format: {sample_id: [exemplar_frame_ids]}

        propagate_annotations(
            view=ctx.target_view(),
            exemplar_frame_field=stored_exemplar_frame_field,
            annotation_field=annotation_field,
            exemplar_assignments=exemplar_assignments,
        )
        return {}

    def resolve_input(self, ctx):
        inputs = types.Object()

        # Check if there are any extract_exemplar_frames runs
        available_runs = ctx.dataset.list_runs()
        available_runs = [
            rr
            for rr in available_runs
            if hasattr(ctx.dataset.get_run_info(rr).config, "exemplar_frame_field")
        ]

        if len(available_runs) > 0:
            run_dropdown = types.Dropdown()
            for run_key in available_runs:
                try:
                    run_info = ctx.dataset.get_run_info(run_key)
                    # Create a label from the run info
                    timestamp = run_info.get("timestamp", "unknown")
                    exemplar_field = run_info.get("config", {}).get("exemplar_frame_field", "unknown")
                    # Format timestamp for display (just date and time, no microseconds)
                    if timestamp != "unknown":
                        timestamp = timestamp.split(".")[0].replace("T", " ")
                    label = f"{run_key} ({exemplar_field}, {timestamp})"
                    run_dropdown.add_choice(run_key, label=label)
                except Exception:
                    # Fallback if we can't get run info
                    run_dropdown.add_choice(run_key, label=run_key)
            
            inputs.enum(
                "exemplar_run_key",
                run_dropdown.values(),
                default=available_runs[0] if available_runs else None,
                label="Exemplar Extraction Run Key",
                view=run_dropdown,
                required=True,
            )
        else:
            raise ValueError("No exemplar extraction runs found. Please run 'extract_exemplar_frames' first.")

        inputs.str(
            "annotation_field",
            label="Annotation Field",
            default="human_labels",
            required=True,
        )

        return types.Property(inputs)


def register(p):
    p.register(ExtractExemplarFrames)
    p.register(PropagateAnnotationsFromExemplars)
