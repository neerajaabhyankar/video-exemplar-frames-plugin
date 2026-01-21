# video-exemplar-frames-plugin

Extract exemplar frames from a video dataset. Propagate annotations from exemplar frames.

## Installation

Install the plugin and its dependencies:

```bash
# Install the plugin
fiftyone plugins download https://github.com/neerajaabhyankar/video-exemplar-frames-plugin

# Install dependencies automatically (recommended)
fiftyone plugins requirements @neerajaabhyankar/video-exemplar-frames-plugin --install
```

## Download Pretrained Weights

After installation, download the pretrained weights to the installed package location:

```bash
WEIGHTS_DIR=$(python -c 'import siamfc; import os; print(os.path.join(os.path.dirname(os.path.abspath(siamfc.__file__)), "weights"))')
mkdir -p "$WEIGHTS_DIR"
echo "Downloading weights to" $WEIGHTS_DIR
gdown "https://drive.google.com/uc?id=1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4" -O "$WEIGHTS_DIR/siamfc_alexnet_e50.pth"
```

## Usage

The plugin provides two operators:

1. **extract_exemplar_frames**: Extract exemplar frames from a video dataset
2. **propagate_annotations_from_exemplars**: Propagate annotations from exemplar frames to all frames

See the plugin documentation for detailed usage instructions.
