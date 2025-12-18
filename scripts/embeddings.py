"""
Compute embeddings and visualization for a FiftyOne image dataset using CLIP.

This script loads a FiftyOne dataset and computes embeddings using CLIP-ViT-Base32,
then creates a visualization of the embeddings.
"""

import fiftyone as fo
import fiftyone.brain as fob
from pathlib import Path


def compute_embeddings(
    dataset_name: str = "basketball_frames",
    model: str = "clip-vit-base32-torch",
    embeddings_field: str = "embeddings",
    brain_key: str = "clip_vis",
):
    """
    Compute embeddings and visualization for a dataset using CLIP.
    
    Args:
        dataset_name: Name of the FiftyOne dataset
        model: Model name for embeddings (default: "clip-vit-base32-torch")
        embeddings_field: Field name to store embeddings
        brain_key: Key for storing the brain run results
    
    Returns:
        The dataset with computed embeddings and visualization
    """
    # Load the dataset
    if dataset_name not in fo.list_datasets():
        raise ValueError(f"Dataset '{dataset_name}' not found. Please run frames_dataset.py first.")
    
    dataset = fo.load_dataset(dataset_name)
    print(f"Loaded dataset: {dataset_name}")
    print(f"  Number of samples: {len(dataset)}")
    
    # Compute embeddings and visualization
    print(f"\nComputing embeddings using {model}...")
    print(f"  Embeddings field: {embeddings_field}")
    print(f"  Brain key: {brain_key}")
    
    fob.compute_visualization(
        dataset,
        model=model,
        embeddings=embeddings_field,
        brain_key=brain_key,
    )
    
    print("\nEmbeddings and visualization computed successfully!")
    print(f"  Embeddings stored in field: {embeddings_field}")
    print(f"  Visualization stored with brain key: {brain_key}")
    
    # Print some info about the embeddings
    if len(dataset) > 0:
        sample = dataset.first()
        if embeddings_field in sample.field_names:
            embeddings = sample[embeddings_field]
            if embeddings is not None:
                print(f"  Embedding dimension: {len(embeddings)}")
    
    return dataset


def compute_visualizations_existing_embeddings(
    dataset: fo.Dataset,
    embeddings_field: str = "embeddings",
    brain_key: str = "clip_vis",
):
    """
    Compute visualizations for a dataset with existing embeddings.
    """
    pass


if __name__ == "__main__":
    # Compute embeddings for the basketball frames dataset
    dataset = compute_embeddings(
        dataset_name="basketball_frames",
        model="clip-vit-base32-torch",
        embeddings_field="embeddings",
        brain_key="clip_vis",
    )
    
    # Print dataset info
    print("\n" + "="*50)
    print("Embeddings computation complete!")
    print("="*50)
    print(f"Dataset: {dataset.name}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Sample fields: {dataset.get_field_schema()}")
    
    # Show brain runs
    if dataset.has_brain_runs:
        print(f"\nBrain runs: {list(dataset.list_brain_runs())}")

