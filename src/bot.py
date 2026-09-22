"""Telegram bot front-end for the digit classifier.

Features:
- Predict a handwritten digit from a photo
- /compare  – run all available models side-by-side
- /explain  – attach a Grad-CAM heatmap

Run from the project root:
    python -m src.bot
"""

import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from . import config
from .preprocessing import make_debug_image, preprocess_image
from .xai import make_gradcam_heatmap, overlay_heatmap_on_digit

TOKEN = config.TELEGRAM_BOT_TOKEN
if not TOKEN:
    raise SystemExit("No bot token provided. Set TELEGRAM_BOT_TOKEN and try again.")

AVAILABLE_MODELS = config.AVAILABLE_MODELS

current_model_name = config.DEFAULT_MODEL
model = None
_model_cache = {}


def get_or_load_model(name: str):
    """Load a model by name (cached)."""
    if name in _model_cache:
        return _model_cache[name]

    filename = AVAILABLE_MODELS.get(name)
    if not filename:
        return None

    path = os.path.join(config.MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    print(f"Loading model: {filename}")
    loaded = tf.keras.models.load_model(path)
    _model_cache[name] = loaded
    print(f"Model '{name}' loaded successfully!")
    return loaded


def load_model(name: str) -> bool:
    """Set the active model."""
    global model, current_model_name
    loaded = get_or_load_model(name)
    if loaded is None:
        return False
    model = loaded
    current_model_name = name
    return True


if not load_model(config.DEFAULT_MODEL):
    print("ERROR: Could not load default model!")
    raise SystemExit(1)


def _image_to_buffer(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


async def predict_and_reply(update: Update, pil_image: Image.Image, compare: bool, explain: bool):
    model_input, canvas_28 = preprocess_image(pil_image)

    if model_input is None:
        await update.message.reply_text(
            "I couldn't clearly find a digit in that photo. Try:\n"
            "- Writing bigger / thicker\n"
            "- Better, even lighting (avoid shadows)\n"
            "- A plain background\n"
            "- Filling more of the frame with the digit"
        )
        return

    debug_img = make_debug_image(canvas_28)

    if not compare:
        prediction = model.predict(model_input, verbose=0)[0]
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        await update.message.reply_photo(
            photo=_image_to_buffer(debug_img),
            caption=(
                f"Predicted: {predicted_digit}\n"
                f"Confidence: {confidence:.2f}%\n"
                f"Model: {current_model_name}\n\n"
                "(This is exactly what the model saw — if it looks wrong, "
                "that's a preprocessing issue; if it looks right but the "
                "prediction is wrong, that's a model issue.)"
            ),
        )

        if explain:
            heatmap, pred_index = make_gradcam_heatmap(
                model_input, model, pred_index=predicted_digit
            )
            gradcam_img = overlay_heatmap_on_digit(canvas_28, heatmap)
            await update.message.reply_photo(
                photo=_image_to_buffer(gradcam_img),
                caption=(
                    "Grad-CAM explanation: warmer colors (red/yellow) show "
                    f"the regions the '{current_model_name}' model relied on "
                    f"most to predict {predicted_digit}."
                ),
            )
        return

    # Compare mode
    lines = []
    for name in AVAILABLE_MODELS:
        m = get_or_load_model(name)
        if m is None:
            lines.append(f"• {name}: not available (model file missing)")
            continue
        prediction = m.predict(model_input, verbose=0)[0]
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        lines.append(f"• {name}: {predicted_digit}  ({confidence:.2f}%)")

    caption = "Compare mode results:\n\n" + "\n".join(lines)
    await update.message.reply_photo(photo=_image_to_buffer(debug_img), caption=caption)

    if explain:
        prediction = model.predict(model_input, verbose=0)[0]
        predicted_digit = int(np.argmax(prediction))
        heatmap, _ = make_gradcam_heatmap(
            model_input, model, pred_index=predicted_digit
        )
        gradcam_img = overlay_heatmap_on_digit(canvas_28, heatmap)
        await update.message.reply_photo(
            photo=_image_to_buffer(gradcam_img),
            caption=(
                f"Grad-CAM for the active model ('{current_model_name}'), "
                f"which predicted {predicted_digit}."
            ),
        )


# -------------------------
# Handlers
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    compare_on = context.user_data.get("compare_mode", False)
    explain_on = context.user_data.get("explain_mode", False)
    text = (
        "Hello!\n\n"
        "Send me a photo of a handwritten digit (0-9).\n\n"
        f"Active model: {current_model_name}\n"
        f"Compare mode: {'ON' if compare_on else 'OFF'}\n"
        f"Explain mode (Grad-CAM): {'ON' if explain_on else 'OFF'}\n\n"
        "Commands:\n"
        "/models  – Show available models\n"
        "/use_best – Switch to the best model\n"
        "/use_final – Switch to the final model\n"
        "/compare – Toggle comparing all models\n"
        "/explain – Toggle Grad-CAM explanations"
    )
    await update.message.reply_text(text)


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Best model", callback_data="use_best")],
        [InlineKeyboardButton("Final model", callback_data="use_final")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    text = (
        "Available models:\n\n"
        "• best  → best_digit_model.keras\n"
        "• final → digit_model.keras\n\n"
        f"Currently active: {current_model_name}"
    )
    await update.message.reply_text(text, reply_markup=reply_markup)


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    name = query.data.replace("use_", "")
    success = load_model(name)

    if success:
        await query.edit_message_text(f"Switched to model: {name}")
    else:
        await query.edit_message_text("Failed to load that model.")


async def use_best(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if load_model("best"):
        await update.message.reply_text("Now using best model")
    else:
        await update.message.reply_text("Could not load best model")


async def use_final(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if load_model("final"):
        await update.message.reply_text("Now using final model")
    else:
        await update.message.reply_text("Could not load final model")


async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current = context.user_data.get("compare_mode", False)
    context.user_data["compare_mode"] = not current
    if not current:
        await update.message.reply_text(
            "Compare mode ON. Send a photo and I'll run it through all "
            "available models and show each result."
        )
    else:
        await update.message.reply_text(
            f"Compare mode OFF. Back to using the active model: {current_model_name}"
        )


async def explain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current = context.user_data.get("explain_mode", False)
    context.user_data["explain_mode"] = not current
    if not current:
        await update.message.reply_text(
            "Explain mode ON. I'll attach a Grad-CAM heatmap showing which "
            "pixels drove each prediction."
        )
    else:
        await update.message.reply_text("Explain mode OFF.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        compare = context.user_data.get("compare_mode", False)
        explain = context.user_data.get("explain_mode", False)
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        image = Image.open(io.BytesIO(photo_bytes))
        await predict_and_reply(update, image, compare, explain)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    if document.mime_type and document.mime_type.startswith("image/"):
        try:
            compare = context.user_data.get("compare_mode", False)
            explain = context.user_data.get("explain_mode", False)
            file = await document.get_file()
            photo_bytes = await file.download_as_bytearray()
            image = Image.open(io.BytesIO(photo_bytes))
            await predict_and_reply(update, image, compare, explain)
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
    else:
        await update.message.reply_text("Please send an image.")


def main():
    request = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )

    app = ApplicationBuilder().token(TOKEN).request(request).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(CommandHandler("use_best", use_best))
    app.add_handler(CommandHandler("use_final", use_final))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(CommandHandler("explain", explain_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()