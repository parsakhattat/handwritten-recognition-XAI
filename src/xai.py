"""Grad-CAM explanations for the digit classifier.

Grad-CAM (Selvaraju et al., 2017) produces a heatmap highlighting which
pixels most influenced a prediction by weighting the last convolutional
layer's feature maps with the gradients of the predicted class score.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.cm as cm


def find_last_conv_layer_name(model: keras.Model) -> str:
    """Return the name of the last Conv2D layer."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found — Grad-CAM requires one.")


def make_gradcam_heatmap(
    model_input: np.ndarray,
    model: keras.Model,
    last_conv_layer_name: str = None,
    pred_index: int = None,
):
    """Compute a Grad-CAM heatmap for a single input.

    Returns (heatmap, pred_index) where heatmap is a 2D array in [0, 1]
    and pred_index is the class the heatmap explains.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)

    x = tf.convert_to_tensor(model_input, dtype=tf.float32)
    conv_output = None

    # Replay the forward pass layer-by-layer so Sequential models
    # loaded via load_model work without a symbolic graph.
    with tf.GradientTape() as tape:
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                tape.watch(x)
                conv_output = x

        predictions = x
        if conv_output is None:
            raise ValueError(
                f"Layer '{last_conv_layer_name}' not found while replaying layers."
            )
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), pred_index


def overlay_heatmap_on_digit(
    canvas_28: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    output_size: int = 140,
) -> Image.Image:
    """Blend Grad-CAM heatmap over the 28×28 digit and upscale for display."""
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (28, 28), Image.BILINEAR
    )
    heatmap_resized = np.array(heatmap_img).astype("float32") / 255.0

    colormap = cm.get_cmap("jet")
    colored = colormap(heatmap_resized)[:, :, :3]
    colored = (colored * 255).astype("uint8")

    base_rgb = np.stack([canvas_28] * 3, axis=-1).astype("uint8")
    blended = (
        base_rgb.astype("float32") * (1 - alpha)
        + colored.astype("float32") * alpha
    )
    blended = np.clip(blended, 0, 255).astype("uint8")

    return Image.fromarray(blended).resize((output_size, output_size), Image.NEAREST)