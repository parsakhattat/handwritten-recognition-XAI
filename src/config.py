"""Central configuration: paths, model registry, and secrets."""

import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MNIST_PATH = os.path.join(DATA_DIR, "mnist.npz")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_digit_model.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

AVAILABLE_MODELS = {
    "best": "best_digit_model.keras",
    "final": "digit_model.keras",
}

DEFAULT_MODEL = "best"

# Set as environment variable before running the bot:
#   export TELEGRAM_BOT_TOKEN="your-token-here"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_BOT_TOKEN:
    TELEGRAM_BOT_TOKEN = input(
        "TELEGRAM_BOT_TOKEN is not set. Paste your bot token to continue "
        "(or Ctrl+C to quit and export TELEGRAM_BOT_TOKEN=... instead): "
    ).strip()

SEED = 42
