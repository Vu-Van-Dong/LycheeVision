import json
import os
import sys
import numpy as np

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import MODEL_PATH, CLASS_INDICES_PATH, IMG_SIZE


CLASS_VIETNAMESE = {
    "Anthrax Leaf": "Lá bị bệnh thán thư",
    "Bituminous Leaf": "Lá bị ứ nhựa / sẫm màu",
    "Curl Leaf": "Lá bị xoăn",
    "Deficiency Leaf": "Lá thiếu dinh dưỡng",
    "Dry Leaf": "Lá khô",
    "Felt Leaf": "Lá bị lông nhung",
    "Fungal Leaf Spot": "Lá bị đốm nấm",
    "Healthy Leaf": "Lá khỏe mạnh",
    "Leaf Blight": "Lá bị cháy lá",
    "Leaf Gall": "Lá bị u / sần lá",
}


CLASS_DESCRIPTIONS = {
    "Anthrax Leaf": "Lá xuất hiện các vết đốm nâu hoặc đen, có thể lan rộng và làm hoại tử mô lá.",
    "Bituminous Leaf": "Lá có biểu hiện sẫm màu hoặc xuất hiện mảng đen bất thường trên bề mặt.",
    "Curl Leaf": "Lá bị cong, cuộn hoặc biến dạng so với lá bình thường.",
    "Deficiency Leaf": "Lá có biểu hiện vàng, nhạt màu hoặc phát triển không đều do thiếu dinh dưỡng.",
    "Dry Leaf": "Lá bị khô, cháy mép hoặc chuyển sang màu nâu vàng.",
    "Felt Leaf": "Lá xuất hiện lớp lông nhung hoặc mảng bất thường trên bề mặt.",
    "Fungal Leaf Spot": "Lá có nhiều đốm nhỏ do nấm, thường có màu nâu, đen hoặc xám.",
    "Healthy Leaf": "Lá có màu xanh tự nhiên, không có dấu hiệu bệnh rõ ràng.",
    "Leaf Blight": "Lá xuất hiện vùng cháy, vàng nâu hoặc hoại tử ở mép lá hay đầu lá.",
    "Leaf Gall": "Lá xuất hiện các nốt u, sần hoặc biến dạng cục bộ.",
}


def load_class_names():
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    index_to_class = {index: name for name, index in class_indices.items()}
    class_names = [index_to_class[i] for i in range(len(index_to_class))]

    return class_names


def load_prediction_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return load_model(MODEL_PATH)


def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img, img_array


def predict_uploaded_image(model, uploaded_file):
    class_names = load_class_names()

    original_img, img_array = preprocess_uploaded_image(uploaded_file)

    predictions = model.predict(img_array)[0]

    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])
    class_name = class_names[class_index]

    return {
        "image": original_img,
        "class_name": class_name,
        "class_name_vi": CLASS_VIETNAMESE.get(class_name, class_name),
        "description": CLASS_DESCRIPTIONS.get(class_name, "Chua co mo ta."),
        "confidence": confidence,
        "all_predictions": predictions,
        "class_names": class_names,
    }