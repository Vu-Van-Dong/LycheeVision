


import streamlit as st
import pandas as pd
import base64

from utils import load_prediction_model, predict_uploaded_image


# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="LycheeVision", 
    page_icon="\U0001F331", 
    layout="wide"
)


# ======================
# LOAD LOGO BASE64
# ======================
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


logo_base64 = get_base64("assets/logo.png")


# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>

/* ================= HEADER CARD ================= */

.header-card{
    
    display:flex;
    
    align-items:center;
    
    gap:25px;
    
    background:white;
    
    padding:28px;
    
    border-radius:20px;
    
    box-shadow:0 8px 25px rgba(0,0,0,0.08);
    
    border-left:8px solid #2E7D32;
    
    margin-bottom:28px;
    
}

.header-logo{
    
    flex-shrink:0;
    
}

.header-logo img{
    
    width:95px;
    
    height:95px;
    
    border-radius:16px;
    
}

.header-content{
    
    flex:1;
    
}

.header-school{
    
    font-size:16px;
    
    font-weight:700;
    
    color:#2E7D32;
    
    text-transform:uppercase;
    
    letter-spacing:1px;
    
    margin-bottom:8px;
    
}

.header-title{

    font-size:34px;

    font-weight:800;

    color:#1E1E1E;

    line-height:1.25;

    margin-bottom:10px;

}

.header-subtitle{

    font-size:17px;

    color:#666;

    line-height:1.5;

}

.stApp{
    background-color:#f4f8f5;
}

/* HEADER */
.main-header{
    background: linear-gradient(135deg,#1b5e20,#43a047);
    padding:25px;
    border-radius:18px;
    display:flex;
    align-items:center;
    gap:20px;
    color:white;
    margin-bottom:25px;
    box-shadow:0 5px 15px rgba(0,0,0,0.15);
}

.main-header img{
    width:90px;
    height:90px;
    border-radius:12px;
}

.main-header h1{
    margin:0;
    font-size:28px;
}

.main-header p{
    margin:5px 0;
    font-size:15px;
    opacity:0.95;
}

/* CARD */
.custom-card{
    background:white;
    padding:20px;
    border-radius:16px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
    margin-bottom:15px;
}

/* INFO BOX */
.info-box{
    background:#eef8f0;
    border-left:5px solid #2e7d32;
    padding:15px;
    border-radius:10px;
    margin-top:12px;
}

/* IMAGE */
img{
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)


import streamlit as st

# ================= CSS =================
st.markdown("""
<style>


div[data-testid="stHorizontalBlock"]{
    align-items:flex-start !important;
}

.left-bar{
    width:6px;
    height:90px;
    background:#2d6f90;
    border-radius:20px;
    margin:auto;
}

.school{
    font-size:15px;
    font-weight:700;
    color:#444;
    margin-bottom:2px;
}

.title{
    font-size:34px;
    font-weight:800;
    color:#111;
    line-height:1.2;
}

.subtitle{
    font-size:16px;
    color:#777;
}
div[data-testid="stVerticalBlockBorderWrapper"]{
    border-left:8px solid #2E7D32 !important;
    border-radius:20px !important;
    box-shadow:0 6px 18px rgba(0,0,0,.08);
}

</style>
""", unsafe_allow_html=True)


# ================= HEADER =================

with st.container(border=True):

    col1, col2 = st.columns([1.2, 8], vertical_alignment="center")

    with col1:
        st.image("assets/logo.png", width=110)

    with col2:

        st.markdown("""
        <div style="
            color:#1B5E20;
            font-size:22px;
            font-weight:700;
            letter-spacing:0.8px;
            margin-bottom:4px;">
            TRƯỜNG ĐẠI HỌC THỦY LỢI
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            font-size:30px;
            font-weight:800;
            line-height:1.2;
            color:#1E1E1E;
            margin-bottom:6px;">
            LycheeVision – Hệ thống ứng dụng CNN trong phát hiện và phân loại sâu bệnh trên cây vải thiều
        </div>
        """, unsafe_allow_html=True)
# ======================
# LOAD MODEL
# ======================
model = load_prediction_model()

if model is None:
    st.error("\u274C Không tìm thấy model. Vui lòng train trước.")
    st.stop()


# ======================
# UPLOAD SECTION
# ======================

st.markdown("""
<div class="custom-card">
    <h3>\U0001F4E4 Tải ảnh lá vải thiều</h3>
    <p>Chọn ảnh để hệ thống phân tích và dự đoán bệnh.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])


# ======================
# DISEASE INFO
# ======================
disease_info = {
    "Anthrax Leaf": {
        "symptom": "Đốm nâu/đen lan rộng gây hoại tử.",
        "treatment": "Phun Copper, Mancozeb, Carbendazim; cắt lá bệnh."
    },
    "Bituminous Leaf": {
        "symptom": "Lá sẫm màu do tiết nhựa bất thường.",
        "treatment": "Diệt rệp, bọ trĩ; dùng Neem oil hoặc thuốc Imidacloprid."
    },
    "Curl Leaf": {
        "symptom": "Lá xoăn, biến dạng.",
        "treatment": "Phun Abamectin; bổ sung vi lượng."
    },
    "Deficiency Leaf": {
        "symptom": "Lá vàng do thiếu dinh dưỡng.",
        "treatment": "Bón NPK + Mg, Zn, Fe."
    },
    "Dry Leaf": {
        "symptom": "Lá khô, cháy mép.",
        "treatment": "Tưới nước hợp lý, bổ sung Kali + Canxi."
    },
    "Felt Leaf": {
        "symptom": "Nhện lông nhung gây mảng bất thường.",
        "treatment": "Phun Abamectin, dầu khoáng."
    },
    "Fungal Leaf Spot": {
        "symptom": "Đốm nấm nâu/đen.",
        "treatment": "Phun Mancozeb, Chlorothalonil."
    },
    "Healthy Leaf": {
        "symptom": "Lá khỏe mạnh bình thường.",
        "treatment": "Duy trì chăm sóc tốt."
    },
    "Leaf Blight": {
        "symptom": "Cháy mép lá, vàng nâu.",
        "treatment": "Phun Copper hydroxide; giảm đạm."
    },
    "Leaf Gall": {
        "symptom": "Lá bị u, sần.",
        "treatment": "Cắt bỏ lá bệnh, dùng thuốc trừ sâu."
    }
}


# ======================
# PREDICTION (dự đoán )
# ======================

if uploaded_file is not None:

    result = predict_uploaded_image(model, uploaded_file)

    # Chia 2 cột
    col1, col2 = st.columns([1, 1.2], gap="large")

    # ================= ẢNH =================
    with col1:
        
        st.image(
            result["image"],
            caption="\U0001F4F7 Ảnh đầu vào",
            width=420
)

    # ================= KẾT QUẢ =================
    with col2:

        confidence_percent = result["confidence"] * 100

        st.markdown("## \U0001F4CA Kết quả")

        if result["class_name"] == "Unknown":

            st.error("\u274C Không phải lá vải hoặc không chắc chắn")
            st.metric("Độ tin cậy", f"{confidence_percent:.2f}%")

        else:

            st.success("\u2705 Dự đoán thành công")

            st.write("**Tên lớp:**", result["class_name"])
            st.write("**Độ tin cậy:**", f"{confidence_percent:.2f}%")

            st.info(result["description"])

            if result["class_name"] in disease_info:

                info = disease_info[result["class_name"]]

                st.markdown("### \U0001F331 Biểu hiện bệnh")
                st.write(info["symptom"])

                
                st.markdown("### \U0001F6E0\uFE0F Cách khắc phục")
                st.write(info["treatment"])

    # ======================
    # PROBABILITY TABLE (bảng xác suất)
    # ======================
    st.markdown("---")
    st.markdown("### \U0001F4C8 Xác suất từng lớp")

    df = pd.DataFrame({
        "Lớp": result["class_names"],
        "Xác suất (%)": [round(x * 100, 2) for x in result["all_predictions"]],
    }).sort_values(by="Xác suất (%)", ascending=False)

    st.dataframe(df, use_container_width=True)

else:
    st.info("\U0001F4CC Vui lòng upload ảnh để bắt đầu")
