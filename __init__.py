"""Video Exemplar Frames plugin.

Extract exemplar frames from a video dataset and propagate annotations.

| Copyright 2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`
"""

import os
import logging
from datetime import datetime

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone.core.utils import add_sys_path

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from exemplars import extract_exemplar_frames
    from annoprop import propagate_annotations

logger = logging.getLogger(__name__)

if not logger.handlers:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    log_file = os.path.join(log_dir, "debug.log")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


class ExtractExemplarFrames(foo.Operator):
    version = "1.0.0"
    
    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="extract_exemplar_frames", 
            label="Extract Exemplar Frames Operator",
            description="Extract exemplar frames from a video dataset",
            icon="/assets/keyframes.svg",
            dynamic=False,
            execution_options=foo.ExecutionOptions(
                allow_immediate=False,
                allow_delegation=True,
                default_choice_to_delegated=True,
            ),
        )

    def resolve_input(self, ctx) -> types.Property:
        inputs = types.Object()
        
        # # Warn for large datasets
        # if len(ctx.view) > 1000:
        #     inputs.view("warning", types.Warning(
        #         label="Large Dataset",
        #         description=f"This operation will process {len(ctx.view)} samples and may take a long time."
        #     ))

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
    
    # def resolve_delegation(self, ctx) -> bool:
    #     """Delegate for heavy operations, but allow immediate execution for small datasets."""
    #     # For test contexts or small datasets, allow immediate execution
    #     view = ctx.target_view() if hasattr(ctx, "target_view") else ctx.view
    #     if len(view) < 100:
    #         return False
    #     return True

    def execute(self, ctx) -> dict:
        # Make dataset persistent if possible (required for custom runs)
        try:
            ctx.dataset.persistent = True
        except Exception:
            pass  # Dataset might not support persistent flag
        
        view = ctx.target_view()
        total_samples = len(view)
        max_fraction_exemplars = ctx.params.get("max_fraction_exemplars", 0.1)
        exemplar_frame_field = ctx.params.get("exemplar_frame_field", "exemplar")
        method = ctx.params.get("method", "random")
        
        # Check if field exists and validate/convert its type
        dataset = ctx.dataset
        if exemplar_frame_field in dataset.get_field_schema():
            logger.debug(f"Exemplar frame field exists: {exemplar_frame_field}")
            field_type_name = type(dataset.get_field_schema()[exemplar_frame_field]).__name__
            if field_type_name != "EmbeddedDocumentField":
                logger.debug(f"Deleting exemplar frame field of incorrect type: {exemplar_frame_field}")
                dataset.delete_sample_field(exemplar_frame_field)
        
        # Ensure the exemplar field exists and declare nested fields for proper schema support
        # Use DynamicEmbeddedDocument to allow dynamic fields
        if exemplar_frame_field not in dataset.get_field_schema():
            logger.info(f"Adding exemplar frame field: {exemplar_frame_field}")
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
        
        # TODO(neeraja): Progress reporting

        extract_exemplar_frames(
            view,
            max_fraction_exemplars=max_fraction_exemplars,
            exemplar_frame_field=exemplar_frame_field,
            method=method,
        )
        logger.info(f"Exemplar frames extracted and stored in field '{exemplar_frame_field}'")

        # TODO(neeraja): move this to a separate function
        
        # Generate run key following convention: <namespace>/<operator>/<version>/<timestamp>
        plugin_namespace = "@neerajaabhyankar/video-exemplar-frames-plugin"
        run_key = f"{plugin_namespace}/{self.config.name}/v{self.version}/{datetime.utcnow().isoformat(timespec='seconds')}Z"
        
        # Store custom run (if dataset supports it)
        try:
            # Store config
            run_config = ctx.dataset.init_run()
            run_config.max_fraction_exemplars = max_fraction_exemplars
            run_config.exemplar_frame_field = exemplar_frame_field
            run_config.method = method
            run_config.dataset_id = ctx.dataset.id
            run_config.dataset_name = getattr(ctx.dataset, "name", None)
            run_config.operator = self.config.name
            run_config.version = self.version
            ctx.dataset.register_run(run_key, run_config, overwrite=True)
            
            # Store results
            run_results = ctx.dataset.init_run_results(run_key)
            run_results.message = f"Exemplar frames extracted and stored in field '{exemplar_frame_field}'"
            run_results.samples_processed = total_samples
            run_results.exemplar_frame_field = exemplar_frame_field
            run_results.method = method
            ctx.dataset.save_run_results(run_key, run_results, overwrite=True)
            
            # Maintain latest alias
            latest_key = f"{plugin_namespace}/{self.config.name}/latest"
            try:
                ctx.dataset.rename_run(latest_key, run_key)
            except Exception:
                # First time; register an alias
                run_info = ctx.dataset.get_run_info(run_key)
                ctx.dataset.register_run(latest_key, run_info.config)
            
            logger.info(f"Custom extract_exemplar_frames run stored: {run_key}")
        except Exception as e:
            # Custom runs not available (e.g., in test context or non-persistent dataset)
            # Continue without failing
            pass

        if exemplar_frame_field not in dataset.get_field_schema():
            logger.warning(f"Exemplar frame field {exemplar_frame_field} not found in dataset")
        
        return {
            "message": f"Exemplar frames extracted and stored in field '{exemplar_frame_field}'",
            "samples_processed": total_samples,
            "run_key": run_key,
        }


class PropagateAnnotationsFromExemplars(foo.Operator):
    version = "1.0.0"
    
    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="propagate_annotations_from_exemplars",
            label="Propagate Annotations From Exemplars Operator",
            description="Propagate annotations from exemplar frames to all the frames",
            icon="/assets/keyframes_anno.svg",
            dynamic=True,
            execution_options=foo.ExecutionOptions(
                allow_immediate=False,
                allow_delegation=True,
                default_choice_to_delegated=True,
            ),
        )

    # def resolve_delegation(self, ctx) -> bool:
    #     """Delegate for heavy operations, but allow immediate execution for small datasets."""
    #     # For test contexts or small datasets, allow immediate execution
    #     view = ctx.target_view() if hasattr(ctx, "target_view") else ctx.view
    #     if len(view) < 100:
    #         return False
    #     return True

    def execute(self, ctx) -> dict:
        if not self.validate_input(ctx):
            return {
                "message": "Validation failed",
                "samples_processed": 0,
                "samples_evaluated": 0,
            }
        
        view = ctx.target_view()
        total_samples = len(view)
        exemplar_frame_field = ctx.params.get("exemplar_frame_field", "exemplar")
        input_annotation_field = ctx.params.get("input_annotation_field", "human_labels")
        output_annotation_field = ctx.params.get("output_annotation_field", "human_labels_propagated")
        
        # TODO(neeraja): allow for (re?)computing exemplar assignments
        # at this stage, given the annotations!

        # TODO(neeraja): progress reporting

        propagation_scores = propagate_annotations(
            view=view,
            exemplar_frame_field=exemplar_frame_field,
            input_annotation_field=input_annotation_field,
            output_annotation_field=output_annotation_field,
        )
        logger.info(f"Annotations propagated from {input_annotation_field} to {output_annotation_field}")
        logger.info(f"Propagation scores available for {len(propagation_scores)} samples")
        if len(propagation_scores) > 0:
            logger.info(f"Mean propagation score: {sum(propagation_scores.values()) / len(propagation_scores)}")
        
        # Generate run key following convention: <namespace>/<operator>/<version>/<timestamp>
        plugin_namespace = "@neerajaabhyankar/video-exemplar-frames-plugin"
        run_key = f"{plugin_namespace}/{self.config.name}/v{self.version}/{datetime.utcnow().isoformat(timespec='seconds')}Z"

        # TODO(neeraja): move this to a separate function

        # Store custom run (if dataset supports it)
        try:
            # Store config
            run_config = ctx.dataset.init_run()
            run_config.exemplar_frame_field = exemplar_frame_field
            run_config.input_annotation_field = input_annotation_field
            run_config.output_annotation_field = output_annotation_field
            run_config.dataset_id = ctx.dataset.id
            run_config.dataset_name = getattr(ctx.dataset, "name", None)
            run_config.operator = self.config.name
            run_config.version = self.version
            ctx.dataset.register_run(run_key, run_config, overwrite=True)
            
            # Store results
            run_results = ctx.dataset.init_run_results(run_key)
            run_results.message = f"Annotations propagated from {input_annotation_field} to {output_annotation_field}"
            run_results.samples_processed = total_samples
            run_results.propagation_scores = propagation_scores
            run_results.num_evaluated = len(propagation_scores)
            if propagation_scores:
                run_results.mean_score = sum(propagation_scores.values()) / len(propagation_scores)
            ctx.dataset.save_run_results(run_key, run_results, overwrite=True)
            
            # Maintain latest alias
            latest_key = f"{plugin_namespace}/{self.config.name}/latest"
            try:
                ctx.dataset.rename_run(latest_key, run_key)
            except Exception:
                # First time; register an alias
                run_info = ctx.dataset.get_run_info(run_key)
                ctx.dataset.register_run(latest_key, run_info.config)

            logger.info(f"Custom propagate_annotations_from_exemplars run stored: {run_key}")
        except Exception as e:
            # Custom runs not available (e.g., in test context or non-persistent dataset)
            # Continue without failing
            pass
        
        return {
            "propagation_score": propagation_scores,
            "message": f"Annotations propagated from {input_annotation_field} to {output_annotation_field}",
            "samples_processed": total_samples,
            "samples_evaluated": len(propagation_scores),
            "run_key": run_key,
        }

    def resolve_input(self, ctx) -> types.Property:
        inputs = types.Object()

        # Get available fields from dataset schema for autocomplete
        schema = ctx.dataset.get_field_schema()
        field_choices = [types.Choice(label=f, value=f) for f in schema.keys()]
        
        inputs.str(
            "exemplar_frame_field",
            label="Exemplar Frame Information Field",
            default="exemplar",
            view=types.AutocompleteView(choices=field_choices) if field_choices else None,
            required=True,
        )

        inputs.str(
            "input_annotation_field",
            label="Annotation Field to Propagate from",
            default="human_labels",
            view=types.AutocompleteView(choices=field_choices) if field_choices else None,
            required=True,
        )

        inputs.str(
            "output_annotation_field",
            label="Annotation Field to Propagate to",
            default="human_labels_propagated",
            required=True,
        )

        return types.Property(inputs)

    def validate_input(self, ctx) -> bool:
        exemplar_frame_field = ctx.params.get("exemplar_frame_field", "exemplar")
        input_annotation_field = ctx.params.get("input_annotation_field", "human_labels")
        output_annotation_field = ctx.params.get("output_annotation_field", "human_labels_propagated")

        if output_annotation_field == input_annotation_field:
            logger.warning(
                f"Output annotation field '{output_annotation_field}' cannot be the same as "
                f"the input annotation field '{input_annotation_field}'. "
                f"Please choose a different output field name to avoid overwriting the source annotations."
            )
            return False
        
        schema = ctx.dataset.get_field_schema()
        if exemplar_frame_field not in schema:
            logger.warning(
                f"Exemplar frame field '{exemplar_frame_field}' not found in the dataset. "
                f"Please run 'extract_exemplar_frames' first."
            )
            return False
        elif (type(schema[exemplar_frame_field]) != fo.EmbeddedDocumentField) or \
            ("is_exemplar" not in schema[exemplar_frame_field].get_field_schema()) or \
            (type(schema[exemplar_frame_field].get_field_schema()["is_exemplar"]) != fo.BooleanField) or \
            ("exemplar_assignment" not in schema[exemplar_frame_field].get_field_schema()):
            logger.warning(
                f"Exemplar frame field '{exemplar_frame_field}' is not of the correct type. "
                f"Please run 'extract_exemplar_frames' first."
            )
            return False
        
        if input_annotation_field not in schema:
            logger.warning(
                f"Input annotation field '{input_annotation_field}' not found in the dataset. "
                f"Please ensure the field exists and contains annotations."
            )
            return False
        
        return True

def register(p):
    p.register(ExtractExemplarFrames)
    p.register(PropagateAnnotationsFromExemplars)
