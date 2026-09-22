# Handwritten Digit Recognition + XAI

A CNN-based handwritten digit classifier trained on MNIST, paired with **Grad-CAM** explanations and a Telegram bot that lets you test real phone photos end-to-end.

The goal is simple: take a photo of a digit you wrote on paper, get a prediction, and see *why* the model made that decision.

---

## Features

- **CNN classifier** — two residual-style conv blocks with BatchNorm and Dropout, trained with data augmentation tailored for real-world photos
- **Robust preprocessing** — automatic polarity detection, Otsu thresholding, bounding-box crop, and center-of-mass alignment so phone photos match MNIST statistics
- **Grad-CAM explanations** — visual heatmaps that highlight which pixels the model relied on most
- **Telegram bot**
  - Send a photo → get prediction + confidence
  - `/compare` — run every available model on the same input
  - `/explain` — attach a Grad-CAM heatmap
  - Switch between *best* and *final* checkpoints on the fly

---

## Model Architecture

| Layer                          | Details                          |
|--------------------------------|----------------------------------|
| Input                          | 28 × 28 × 1                      |
| Conv2D × 2 + BN + MaxPool      | 32 filters, 3×3, Dropout 0.25    |
| Conv2D × 2 + BN + MaxPool      | 64 filters, 3×3, Dropout 0.25    |
| Flatten → Dense 128 + BN       | Dropout 0.4                      |
| Softmax                        | 10 classes                       |

The last convolutional layer is explicitly named `last_conv` so Grad-CAM can locate it reliably.

**Training setup**
- Optimizer: Adam (lr = 0.001)
- Loss: categorical cross-entropy
- Augmentation: rotation ±12°, shift ±12%, zoom ±12%, shear 8°
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (best val accuracy)
- Typical result: ~99% test accuracy after ~30–40 epochs

---

## Setup

### 1. Clone & install

    git clone https://github.com/parsakhattat/handwritten-recongition-XAI.git
    cd handwritten-recongition-XAI
    python -m venv venv
    source venv/bin/activate          # Windows: venv\Scripts\activate
    pip install -r requirements.txt

### 2. Run the Telegram bot

Pre-trained models and the MNIST dataset are already included in the repository.

1. Create a bot via [@BotFather](https://t.me/BotFather) and copy the token.
2. Export it and start the bot:

    export TELEGRAM_BOT_TOKEN="your-token-here"
    python -m src.bot

If the environment variable is missing, the bot will prompt you interactively (the token is never written to disk).

### 3. (Optional) Retrain the model

If you want to retrain from scratch:

    python -m src.train

This will overwrite the checkpoints in the `models/` folder and regenerate `training_history.png`.

---

## Bot Commands

| Command       | Description                                      |
|---------------|--------------------------------------------------|
| `/start`      | Show status and available commands               |
| `/models`     | List models + inline buttons to switch           |
| `/use_best`   | Switch to the best-validation checkpoint         |
| `/use_final`  | Switch to the final-epoch checkpoint             |
| `/compare`    | Toggle side-by-side comparison of all models     |
| `/explain`    | Toggle Grad-CAM heatmap on every prediction      |

Just send a photo (or an image document) of a handwritten digit.

---

## Preprocessing Pipeline

Real phone photos differ a lot from the clean MNIST digits. The pipeline tries to close that gap:

1. Convert to grayscale
2. Auto-detect background polarity (light paper vs dark paper)
3. Light Gaussian blur
4. Otsu adaptive threshold
5. Locate & crop the digit region
6. Resize to ≤ 20 × 20 and paste onto a 28 × 28 black canvas
7. Re-center by **center of mass** (same method used in the original MNIST pipeline)

The bot also returns a debug image so you can see exactly what the model received.

---

## Explainability (Grad-CAM)

When explain mode is on, the bot computes a Grad-CAM heatmap over the last convolutional layer and overlays it on the preprocessed digit.

- **Warmer colors** (red / yellow) → pixels that most increased the predicted class score
- **Cooler colors** (blue) → less influential regions

No architecture changes or retraining are required; Grad-CAM works on any CNN that has at least one Conv2D layer.

---

## Results

Typical training curves reach > 99 % validation accuracy within the first 10–15 epochs and stay stable. The final test accuracy is usually in the **99.0 – 99.4 %** range depending on the random seed and augmentation strength.

Because the model is deliberately trained with stronger augmentation than classic MNIST benchmarks, it generalizes better to the noisy, rotated, and imperfectly lit digits that appear in real photos.

---

## Requirements

- Python 3.9+
- TensorFlow 2.15+
- Pillow, NumPy, Matplotlib
- python-telegram-bot 21+

See `requirements.txt` for exact pins.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

- MNIST dataset (LeCun et al.)
- Grad-CAM (Selvaraju et al., 2017)
- TensorFlow / Keras

---

*Built as a portfolio project combining practical computer vision, robust preprocessing, and modern explainable AI techniques.*
