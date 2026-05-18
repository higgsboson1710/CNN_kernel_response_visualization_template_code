# 🧠 VGG16 Feature Map Visualizer

Ever wondered what a neural network actually *sees* when it looks at an image? This project cracks open VGG16 — one of the most well-known deep convolutional neural networks — and visualizes what's happening inside it, layer by layer.

---

## What This Project Does

At a high level, this notebook does three things:

1. **Loads a pretrained VGG16 model** and inspects its convolutional layers
2. **Visualizes the learned filters** from the first conv layer — these are the raw "edge detectors" and "texture finders" the network has learned from training on ImageNet
3. **Runs a real image through the network** and captures the feature maps at multiple depths — showing how the network progressively transforms a photo into abstract representations

---

## How It Works

### Step 1 — Filter Visualization
The very first convolutional layer of VGG16 has 64 filters, each with 3 channels (for RGB). We extract those filter weights, normalize them to [0, 1], and plot them as grayscale images. Think of each filter as a tiny pattern detector — some respond to horizontal edges, some to diagonal lines, some to color contrasts.

### Step 2 — Single Layer Feature Map
We take the image, pass it through just the first conv layer, and look at all 64 output channels. Each channel is a "view" of the image through one filter's lens — highlighting whatever pattern that filter is tuned to detect.

### Step 3 — Multi-Layer Feature Maps
This is where it gets interesting. We extract outputs from 5 different layers deep inside the network (layers 2, 5, 9, 13, and 17). As you go deeper:
- Early layers → fine textures, edges, colors
- Middle layers → shapes, object parts
- Deep layers → high-level abstractions (barely resembles the original image)

The grid size adapts dynamically since deeper layers have up to 512 filters — no hardcoded 8×8 grids breaking things.

---

## Setup

This notebook runs on **Google Colab** with Drive mounted.

```
Google Drive
└── ML_DATASETS/
    └── CLG.jpeg        ← your input image
```

No extra installs needed — Colab comes with TensorFlow/Keras pre-installed.

---

## Dependencies

| Library | Purpose |
|---|---|
| `keras` / `tensorflow` | VGG16 model + preprocessing |
| `matplotlib` | Plotting filters and feature maps |
| `numpy` | Array manipulation |
| `google.colab` | Drive mounting |

---

## File Structure

```
├── vgg16_visualization.ipynb   # Main notebook
└── README.md                   # This file
```

---

## Key Concepts

**VGG16** — A 16-layer deep CNN originally trained on ImageNet (1.2M images, 1000 classes). We use it here as a feature extractor, not a classifier.

**Feature Map** — The output of a conv layer when an image passes through it. Each filter produces one channel of the feature map.

**Filter Visualization** — Directly plotting the learned weights of a conv layer. Works best on early layers where patterns are still interpretable to the human eye.

---

## Sample Output

Running the notebook produces:
- A 6×3 grid of filter visualizations from Layer 1
- A feature map grid for Layer 1 (64 channels)
- Feature map grids for Layers 2, 5, 9, 13, and 17 — each labeled with layer index and filter count

---

## Notes

- The image is loaded at **224×224** — VGG16's required input size
- Preprocessing follows VGG16's standard (mean subtraction per ImageNet stats)
- Feature maps are displayed in grayscale for clarity — each channel independently shows activation strength

---

*Built as part of ML coursework — exploring how convolutional neural networks build up visual understanding from pixels to patterns.*
