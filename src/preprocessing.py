"""Convert a photo of a handwritten digit into MNIST-format input.

Pipeline: auto-detect polarity → denoise → Otsu threshold → crop →
resize → center by mass. Matches the original MNIST preprocessing so
models trained on MNIST generalize better to real phone photos.
"""

import numpy as np
from PIL import Image, ImageFilter


def otsu_threshold(gray_arr: np.ndarray) -> int:
    """Compute Otsu's threshold for an 8-bit grayscale array."""
    hist, _ = np.histogram(gray_arr.astype(np.uint8), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = gray_arr.size
    sum_total = np.dot(np.arange(256), hist)

    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 128

    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = i

    return threshold


def shift_array(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a 2D array by (dy, dx) without wrapping edges."""
    h, w = arr.shape
    shifted = np.zeros_like(arr)

    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)

    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]

    return shifted


def preprocess_image(pil_image: Image.Image):
    """Convert a photo of a handwritten digit into MNIST-style input.

    Returns (model_input, canvas_28) on success:
      - model_input: shape (1, 28, 28, 1), values in [0, 1]
      - canvas_28:   raw uint8 28×28 grayscale (for debug / XAI overlays)

    Returns (None, None) if no digit could be located.
    """
    img = pil_image.convert("L")
    arr = np.array(img).astype("float32")

    # Auto-detect polarity from border pixels
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if border.mean() > 127:
        arr = 255.0 - arr

    blurred = np.array(
        Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).filter(
            ImageFilter.GaussianBlur(radius=1)
        )
    ).astype("float32")

    threshold = otsu_threshold(blurred)
    threshold = max(threshold, 30)
    mask = blurred > threshold

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None, None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    if (rmax - rmin) < 3 or (cmax - cmin) < 3:
        return None, None

    pad = 2
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(arr.shape[1] - 1, cmax + pad)

    cropped = arr[rmin : rmax + 1, cmin : cmax + 1].copy()
    cropped[cropped < threshold] = 0

    digit_img = Image.fromarray(np.clip(cropped, 0, 255).astype("uint8"))
    digit_img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    x = (28 - digit_img.width) // 2
    y = (28 - digit_img.height) // 2
    canvas.paste(digit_img, (x, y))
    canvas_arr = np.array(canvas).astype("float32")

    # Recenter by center of mass (matches original MNIST pipeline)
    total_mass = canvas_arr.sum()
    if total_mass > 0:
        yy, xx = np.indices(canvas_arr.shape)
        cy = (yy * canvas_arr).sum() / total_mass
        cx = (xx * canvas_arr).sum() / total_mass
        dy = int(round(14 - cy))
        dx = int(round(14 - cx))
        canvas_arr = shift_array(canvas_arr, dy, dx)

    canvas_28 = np.clip(canvas_arr, 0, 255).astype("uint8")
    model_input = (canvas_arr / 255.0).reshape(1, 28, 28, 1)

    return model_input, canvas_28


def make_debug_image(canvas_28: np.ndarray, size: int = 140) -> Image.Image:
    """Upscale the 28×28 canvas so humans can inspect model input."""
    return Image.fromarray(canvas_28).resize((size, size), Image.NEAREST)