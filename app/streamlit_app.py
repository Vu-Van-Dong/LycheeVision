import streamlit as st
import pandas as pd

from utils import load_prediction_model, predict_uploaded_image


st.set_page_config(
    page_title="LycheeVision",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 LycheeVision")
st.subheader("Hệ thống phát hiện và phân loại sâu bệnh trên cây vải thiều")

st.write(
    """
    Ứng dụng sử dụng mô hình **ResNet50 kết hợp Transfer Learning**
    để phân loại tình trạng lá vải thiều từ hình ảnh đầu vào.
    """
)

model = load_prediction_model()

if model is None:
    st.error(
        "Chưa tìm thấy model `models/lychee_resnet50.h5`. "
        "Hãy train model trước bằng lệnh: `python src/train.py`"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Tải ảnh lá vải thiều lên hệ thống",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    result = predict_uploaded_image(model, uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(result["image"], caption="Ảnh đầu vào", use_container_width=True)

    with col2:
        st.markdown("### Kết quả dự đoán")
        st.success(result["class_name_vi"])

        st.write("**Tên lớp:**", result["class_name"])
        st.write("**Độ tin cậy:**", f"{result['confidence'] * 100:.2f}%")

        st.markdown("### Mô tả")
        st.write(result["description"])

    st.markdown("---")
    st.markdown("### Xác suất dự đoán cho từng lớp")

    df = pd.DataFrame({
        "Lớp": result["class_names"],
        "Xác suất (%)": [round(float(x) * 100, 2) for x in result["all_predictions"]],
    })

    df = df.sort_values(by="Xác suất (%)", ascending=False)

    st.dataframe(df, use_container_width=True)

else:
    st.info("Vui lòng tải lên một ảnh để hệ thống dự đoán.")