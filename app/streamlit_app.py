import streamlit as st
import pandas as pd

from utils import load_prediction_model, predict_uploaded_image


# ======================
# CONFIG UI
# ======================
st.set_page_config(
    page_title="LycheeVision",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 LycheeVision")
st.subheader("Phân loại bệnh trên lá vải thiều bằng AI")

st.write(
    """
    Hệ thống AI sử dụng ResNet50 + Transfer Learning
    để nhận diện bệnh trên lá vải thiều.
    """
)

# ======================
# LOAD MODEL
# ======================
model = load_prediction_model()

if model is None:
    st.error("❌ Không tìm thấy model. Hãy train trước.")
    st.stop()

# ======================
# UPLOAD
# ======================
uploaded_file = st.file_uploader(
    "📤 Upload ảnh lá vải",
    type=["jpg", "jpeg", "png", "webp"]
)

# ======================
# DISEASE INFO (THÊM MỚI)
# ======================
disease_info = {
    "Anthrax Leaf": {
        "symptom": "Đốm nâu/đen lan rộng gây hoại tử.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc gốc đồng (Copper), Mancozeb, Carbendazim, giữ vườn thông thoáng."
    },
    "Bituminous Leaf": {
        "symptom": "Lá sẫm màu, đen do tiết nhựa bất thường.",
        "treatment": "Diệt côn trùng chích hút (rệp, bọ trĩ), dùng Imidacloprid hoặc dầu neem"
    },
    "Curl Leaf": {
        "symptom": "Lá bị xoăn, cong, biến dạng.",
        "treatment": "Phun Abamectin, Spinosad; loại bỏ lá bệnh; bổ sung vi lượng."
    },
    "Deficiency Leaf": {
        "symptom": "Lá vàng, thiếu dinh dưỡng.",
        "treatment": "Bón NPK cân đối, bổ sung Mg, Zn, Fe, phun phân bón lá."
    },
    "Dry Leaf": {
        "symptom": "Lá khô, cháy mép.",
        "treatment": "Tưới nước hợp lý, bổ sung Kali + Canxi."
    },
    "Felt Leaf": {
        "symptom": "Lớp lông nhung do nhện hại.",
        "treatment": "Phun Abamectin, Fenpyroximate, dùng dầu khoáng."
    },
    "Fungal Leaf Spot": {
        "symptom": "Đốm nấm nâu/đen/xám.",
        "treatment": "Phun Mancozeb, Chlorothalonil, thu gom lá rụng."
    },
    "Healthy Leaf": {
        "symptom": "Lá khỏe mạnh bình thường.",
        "treatment": "Duy trì chăm sóc, bón phân định kỳ, phòng bệnh."
    },
    "Leaf Blight": {
        "symptom": "Cháy mép lá, vàng nâu.",
        "treatment": "Phun Copper hydroxide, Propineb, giảm bón đạm."
    },
    "Leaf Gall": {
        "symptom": "Lá bị u, sần, biến dạng.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc trừ sâu."
    }
}


# ======================
# PROCESS
# ======================
if uploaded_file is not None:

    result = predict_uploaded_image(model, uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(result["image"], caption="Ảnh đầu vào", use_container_width=True)

    with col2:

        confidence_percent = result["confidence"] * 100

        st.markdown("## 📊 Kết quả")

        # UNKNOWN
        if result["class_name"] == "Unknown":
            st.error("❌ Không phải lá vải hoặc không chắc chắn")
            st.write(f"Độ tin cậy: {confidence_percent:.2f}%")

        # NORMAL
        else:
            st.success("✅ Dự đoán thành công")

            st.write("**Tên lớp:**", result["class_name"])
            st.write("**Độ tin cậy:**", f"{confidence_percent:.2f}%")

            st.info("🧠 " + result["description"])

            # ======================
            # THÊM MÔ TẢ + CÁCH KHẮC PHỤC
            # ======================
            if result["class_name"] in disease_info:
                info = disease_info[result["class_name"]]

                st.markdown("### 🌿 Biểu hiện bệnh")
                st.write(info["symptom"])

                st.markdown("### 🛠️ Cách khắc phục")
                st.write(info["treatment"])

    # ======================
    # TABLE
    # ======================
    st.markdown("---")
    st.markdown("### 📈 Xác suất từng lớp")

    df = pd.DataFrame({
        "Lớp": result["class_names"],
        "Xác suất (%)": [round(x * 100, 2) for x in result["all_predictions"]],
    }).sort_values(by="Xác suất (%)", ascending=False)

    st.dataframe(df, use_container_width=True)

else:
    st.info("📌 Vui lòng upload ảnh để dự đoán")