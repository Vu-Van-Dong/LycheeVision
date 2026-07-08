import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

from tensorflow.keras.applications.resnet50 import preprocess_input


MODEL_PATH = "models/lychee_resnet50.h5"
CLASS_PATH = "models/class_indices.json"

# ======================
# NGƯỠNG ỔN ĐỊNH (QUAN TRỌNG)
# ======================
CONF_THRESHOLD = 0.65
GAP_THRESHOLD = 0.20


# ======================
# LOAD MODEL
# ======================
def load_prediction_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


# ======================
# LOAD CLASS
# ======================
def load_classes():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return list(json.load(f).keys())


# ======================
# MÔ TẢ
# ======================
def get_description(name):
    desc = {
        "Healthy Leaf": "Lá khỏe mạnh.",
        "Deficiency Leaf": "Thiếu dinh dưỡng.",
        "Leaf Blight": "Bệnh cháy lá.",
        "Leaf Gall": "Bệnh u lá.",
        "Fungal Leaf Spot": "Lá bị đốm nấm.",
        "Dry Leaf": "Lá khô.",
        "Anthrax Leaf": "Bệnh thán thư.",
        "Bituminous Leaf": "Lá bị ứ nhựa, sẫm màu.",
        "Curl Leaf": "Lá xoăn.",
        "Felt Leaf": "Lá bị nhện lông nhung."
    }
    return desc.get(name, "")
# ======================
# PREDICT (ỔN ĐỊNH VERSION)
# ======================
def predict_uploaded_image(model, uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")

    img = image.resize((224, 224))
    x = np.array(img)

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)[0]

    # ======================
    # TÍNH ĐỘ TỰ TIN
    # ======================
    confidence = np.max(preds)

    class_names = load_classes()

    top1_idx = np.argmax(preds)
    top1 = preds[top1_idx]

    sorted_preds = np.sort(preds)
    top2 = sorted_preds[-2]

    gap = top1 - top2

    # ======================
    # LOGIC ỔN ĐỊNH
    # ======================
    if top1 < CONF_THRESHOLD or gap < GAP_THRESHOLD:
        return {
            "image": image,
            "class_name": "Unknown",
            "class_name_vi": "Không phải lá vải / Không chắc chắn",
            "confidence": float(confidence),
            "description": "Model không đủ chắc chắn để dự đoán.",
            "all_predictions": preds,
            "class_names": class_names
        }

    class_name = class_names[top1_idx]

    return {
        "image": image,
        "class_name": class_name,
        "class_name_vi": class_name,
        "confidence": float(confidence),
        "description": get_description(class_name),
        "all_predictions": preds,
        "class_names": class_names
    }