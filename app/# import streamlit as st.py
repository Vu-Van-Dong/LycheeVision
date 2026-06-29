# import streamlit as st
# import pandas as pd

# from utils import load_prediction_model, predict_uploaded_image


# # ======================
# # CONFIG UI
# # ======================
# st.set_page_config(
#     page_title="LycheeVision",
#     page_icon="🌿",
#     layout="wide",
# )

# st.title("🌿 LycheeVision")
# st.subheader("Phân loại bệnh trên lá vải thiều bằng AI")

# st.write(
#     """
#     Hệ thống AI sử dụng ResNet50 + Transfer Learning
#     để nhận diện bệnh trên lá vải thiều.
#     """
# )

# # ======================
# # LOAD MODEL
# # ======================
# model = load_prediction_model()

# if model is None:
#     st.error("❌ Không tìm thấy model. Hãy train trước.")
#     st.stop()

# # ======================
# # UPLOAD
# # ======================
# uploaded_file = st.file_uploader(
#     "📤 Upload ảnh lá vải",
#     type=["jpg", "jpeg", "png", "webp"]
# )

# # ======================
# # DISEASE INFO (THÊM MỚI)
# # ======================
# disease_info = {
#     "Anthrax Leaf": {
#         "symptom": "Đốm nâu/đen lan rộng gây hoại tử.",
#         "treatment": "Cắt bỏ lá bệnh, phun thuốc gốc đồng (Copper), Mancozeb, Carbendazim, giữ vườn thông thoáng."
#     },
#     "Bituminous Leaf": {
#         "symptom": "Lá sẫm màu, đen do tiết nhựa bất thường.",
#         "treatment": "Diệt côn trùng chích hút (rệp, bọ trĩ), dùng Imidacloprid hoặc dầu neem"
#     },
#     "Curl Leaf": {
#         "symptom": "Lá bị xoăn, cong, biến dạng.",
#         "treatment": "Phun Abamectin, Spinosad; loại bỏ lá bệnh; bổ sung vi lượng."
#     },
#     "Deficiency Leaf": {
#         "symptom": "Lá vàng, thiếu dinh dưỡng.",
#         "treatment": "Bón NPK cân đối, bổ sung Mg, Zn, Fe, phun phân bón lá."
#     },
#     "Dry Leaf": {
#         "symptom": "Lá khô, cháy mép.",
#         "treatment": "Tưới nước hợp lý, bổ sung Kali + Canxi."
#     },
#     "Felt Leaf": {
#         "symptom": "Lớp lông nhung do nhện hại.",
#         "treatment": "Phun Abamectin, Fenpyroximate, dùng dầu khoáng."
#     },
#     "Fungal Leaf Spot": {
#         "symptom": "Đốm nấm nâu/đen/xám.",
#         "treatment": "Phun Mancozeb, Chlorothalonil, thu gom lá rụng."
#     },
#     "Healthy Leaf": {
#         "symptom": "Lá khỏe mạnh bình thường.",
#         "treatment": "Duy trì chăm sóc, bón phân định kỳ, phòng bệnh."
#     },
#     "Leaf Blight": {
#         "symptom": "Cháy mép lá, vàng nâu.",
#         "treatment": "Phun Copper hydroxide, Propineb, giảm bón đạm."
#     },
#     "Leaf Gall": {
#         "symptom": "Lá bị u, sần, biến dạng.",
#         "treatment": "Cắt bỏ lá bệnh, phun thuốc trừ sâu."
#     }
# }


# # ======================
# # PROCESS
# # ======================
# if uploaded_file is not None:

#     result = predict_uploaded_image(model, uploaded_file)

#     col1, col2 = st.columns([1, 1])

#     with col1:
#         st.image(result["image"], caption="Ảnh đầu vào", use_container_width=True)

#     with col2:

#         confidence_percent = result["confidence"] * 100

#         st.markdown("## 📊 Kết quả")

#         # UNKNOWN
#         if result["class_name"] == "Unknown":
#             st.error("❌ Không phải lá vải hoặc không chắc chắn")
#             st.write(f"Độ tin cậy: {confidence_percent:.2f}%")

#         # NORMAL
#         else:
#             st.success("✅ Dự đoán thành công")

#             st.write("**Tên lớp:**", result["class_name"])
#             st.write("**Độ tin cậy:**", f"{confidence_percent:.2f}%")

#             st.info("🧠 " + result["description"])

#             # ======================
#             # THÊM MÔ TẢ + CÁCH KHẮC PHỤC
#             # ======================
#             if result["class_name"] in disease_info:
#                 info = disease_info[result["class_name"]]

#                 st.markdown("### 🌿 Biểu hiện bệnh")
#                 st.write(info["symptom"])

#                 st.markdown("### 🛠️ Cách khắc phục")
#                 st.write(info["treatment"])

#     # ======================
#     # TABLE
#     # ======================
#     st.markdown("---")
#     st.markdown("### 📈 Xác suất từng lớp")

#     df = pd.DataFrame({
#         "Lớp": result["class_names"],
#         "Xác suất (%)": [round(x * 100, 2) for x in result["all_predictions"]],
#     }).sort_values(by="Xác suất (%)", ascending=False)

#     st.dataframe(df, use_container_width=True)

# else:
#     st.info("📌 Vui lòng upload ảnh để dự đoán")










# import streamlit as st
# import pandas as pd
# from streamlit_option_menu import option_menu
# import plotly.express as px

# from utils import load_prediction_model, predict_uploaded_image


# # ======================
# # PAGE CONFIG
# # ======================
# st.set_page_config(
#     page_title="LycheeVision",
#     page_icon="🌿",
#     layout="wide"
# )

# # ======================
# # CUSTOM CSS
# # ======================
# st.markdown("""
# <style>

# /* Background */
# .stApp{
#     background-color:#f4f8f5;
# }

# /* Header */
# .main-header{
#     background: linear-gradient(135deg,#1b5e20,#43a047);
#     padding:30px;
#     border-radius:18px;
#     text-align:center;
#     color:white;
#     margin-bottom:25px;
#     box-shadow:0 5px 15px rgba(0,0,0,0.15);
# }

# .main-header h1{
#     margin-bottom:5px;
# }

# .main-header p{
#     font-size:16px;
#     opacity:0.95;
# }

# /* Card */
# .custom-card{
#     background:white;
#     padding:20px;
#     border-radius:16px;
#     box-shadow:0 4px 12px rgba(0,0,0,0.08);
#     margin-bottom:15px;
# }

# /* Disease info */
# .info-box{
#     background:#eef8f0;
#     border-left:5px solid #2e7d32;
#     padding:15px;
#     border-radius:10px;
#     margin-top:12px;
# }

# /* Upload */
# [data-testid="stFileUploader"]{
#     background:white;
#     border-radius:15px;
#     padding:10px;
#     box-shadow:0 4px 12px rgba(0,0,0,0.08);
# }

# /* Metrics */
# [data-testid="metric-container"]{
#     background:white;
#     border-radius:12px;
#     padding:10px;
#     box-shadow:0 3px 10px rgba(0,0,0,0.08);
# }

# /* Dataframe */
# [data-testid="stDataFrame"]{
#     border-radius:12px;
# }

# /* Image */
# img{
#     border-radius:12px;
# }

# </style>
# """, unsafe_allow_html=True)


# # ======================
# # HEADER
# # ======================
# st.markdown("""
# <div class="main-header">
#     <h1>LycheeVision</h1>
#     <p>Hệ thống nhận diện bệnh trên lá vải thiều bằng trí tuệ nhân tạo</p>
#     <p>Mô hình ResNet50 Transfer Learning</p>
# </div>
# """, unsafe_allow_html=True)


# # ======================
# # LOAD MODEL
# # ======================
# model = load_prediction_model()

# if model is None:
#     st.error("Không tìm thấy mô hình. Vui lòng huấn luyện mô hình trước.")
#     st.stop()


# # ======================
# # UPLOAD SECTION
# # ======================
# st.markdown("""
# <div class="custom-card">
#     <h3>Tải ảnh lá vải thiều</h3>
#     <p>Chọn ảnh để hệ thống phân tích và dự đoán tình trạng lá cây.</p>
# </div>
# """, unsafe_allow_html=True)

# uploaded_file = st.file_uploader(
#     label="",
#     type=["jpg", "jpeg", "png", "webp"]
# )


# # ======================
# # DISEASE INFORMATION
# # ======================
# disease_info = {
#     "Anthrax Leaf": {
#         "symptom": "Đốm nâu hoặc đen lan rộng trên lá, gây hoại tử mô lá.",
#         "treatment": "Cắt bỏ lá bệnh, sử dụng thuốc gốc đồng (Copper), Mancozeb hoặc Carbendazim, giữ vườn thông thoáng."
#     },
#     "Bituminous Leaf": {
#         "symptom": "Lá chuyển màu đen hoặc sẫm màu do hiện tượng tiết nhựa bất thường.",
#         "treatment": "Kiểm soát côn trùng chích hút bằng Imidacloprid hoặc dầu Neem."
#     },
#     "Curl Leaf": {
#         "symptom": "Lá xoăn, cong và biến dạng.",
#         "treatment": "Sử dụng Abamectin hoặc Spinosad, loại bỏ lá bệnh và bổ sung vi lượng."
#     },
#     "Deficiency Leaf": {
#         "symptom": "Lá vàng do thiếu dinh dưỡng.",
#         "treatment": "Bón phân NPK cân đối, bổ sung Mg, Zn, Fe và phân bón lá."
#     },
#     "Dry Leaf": {
#         "symptom": "Lá khô hoặc cháy mép.",
#         "treatment": "Điều chỉnh chế độ tưới nước, bổ sung Kali và Canxi."
#     },
#     "Felt Leaf": {
#         "symptom": "Xuất hiện lớp lông nhung do nhện hại gây ra.",
#         "treatment": "Phun Abamectin, Fenpyroximate hoặc dầu khoáng."
#     },
#     "Fungal Leaf Spot": {
#         "symptom": "Đốm nâu, đen hoặc xám trên bề mặt lá.",
#         "treatment": "Sử dụng Mancozeb, Chlorothalonil và thu gom lá rụng."
#     },
#     "Healthy Leaf": {
#         "symptom": "Lá khỏe mạnh, không xuất hiện dấu hiệu bệnh.",
#         "treatment": "Duy trì chế độ chăm sóc và phòng bệnh định kỳ."
#     },
#     "Leaf Blight": {
#         "symptom": "Cháy mép lá, xuất hiện vùng vàng nâu.",
#         "treatment": "Sử dụng Copper Hydroxide hoặc Propineb, hạn chế bón thừa đạm."
#     },
#     "Leaf Gall": {
#         "symptom": "Lá xuất hiện u hoặc sần bất thường.",
#         "treatment": "Cắt bỏ lá bệnh và sử dụng thuốc bảo vệ thực vật phù hợp."
#     }
# }


# # ======================
# # PREDICTION
# # ======================
# if uploaded_file is not None:

#     result = predict_uploaded_image(model, uploaded_file)

#     col1, col2 = st.columns([1, 1])

#     with col1:
#         st.image(
#             result["image"],
#             caption="Ảnh đầu vào",
#             use_container_width=True
#         )

#     with col2:

#         confidence_percent = result["confidence"] * 100

#         st.markdown("""
#         <div class="custom-card">
#             <h2>Kết quả nhận diện</h2>
#         </div>
#         """, unsafe_allow_html=True)

#         if result["class_name"] == "Unknown":

#             st.error(
#                 "Không xác định được lá vải hoặc độ tin cậy quá thấp."
#             )

#             st.metric(
#                 "Độ tin cậy",
#                 f"{confidence_percent:.2f}%"
#             )

#             st.progress(float(result["confidence"]))

#         else:

#             st.success("Dự đoán thành công")

#             metric1, metric2 = st.columns(2)

#             with metric1:
#                 st.metric(
#                     "Tên lớp",
#                     result["class_name"]
#                 )

#             with metric2:
#                 st.metric(
#                     "Độ tin cậy",
#                     f"{confidence_percent:.2f}%"
#                 )

#             st.progress(float(result["confidence"]))

#             st.info(result["description"])

#             if result["class_name"] in disease_info:

#                 info = disease_info[result["class_name"]]

#                 st.markdown(f"""
#                 <div class="info-box">
#                     <h4>Biểu hiện bệnh</h4>
#                     <p>{info["symptom"]}</p>
#                 </div>
#                 """, unsafe_allow_html=True)

#                 st.markdown(f"""
#                 <div class="info-box">
#                     <h4>Giải pháp khắc phục</h4>
#                     <p>{info["treatment"]}</p>
#                 </div>
#                 """, unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)

#     # ======================
#     # PROBABILITY TABLE
#     # ======================
#     st.markdown("""
#     <div class="custom-card">
#         <h3>Xác suất dự đoán các lớp</h3>
#     </div>
#     """, unsafe_allow_html=True)

#     df = pd.DataFrame({
#         "Lớp": result["class_names"],
#         "Xác suất (%)": [
#             round(x * 100, 2)
#             for x in result["all_predictions"]
#         ]
#     }).sort_values(
#         by="Xác suất (%)",
#         ascending=False
#     )

#     st.dataframe(
#         df,
#         use_container_width=True,
#         hide_index=True
#     )

#     st.bar_chart(
#         df.set_index("Lớp")
#     )

# else:

#     st.markdown("""
#     <div class="custom-card" style="text-align:center;">
#         <h3>Chưa có hình ảnh</h3>
#         <p>Vui lòng tải ảnh lá vải thiều để bắt đầu nhận diện.</p>
#     </div>
#     """, unsafe_allow_html=True)

















# import streamlit as st
# import pandas as pd
# import base64

# from utils import load_prediction_model, predict_uploaded_image


# # ======================
# # PAGE CONFIG
# # ======================
# st.set_page_config(
#     page_title="LycheeVision",
#     page_icon="🌿",
#     layout="wide"
# )


# # ======================
# # LOAD LOGO BASE64
# # ======================
# def get_base64(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode()


# logo_base64 = get_base64("assets/logo.png")


# # ======================
# # CUSTOM CSS
# # ======================
# st.markdown("""
# <style>

# .stApp{
#     background-color:#f4f8f5;
# }

# /* HEADER */
# .main-header{
#     background: linear-gradient(135deg,#1b5e20,#43a047);
#     padding:25px;
#     border-radius:18px;
#     display:flex;
#     align-items:center;
#     gap:20px;
#     color:white;
#     margin-bottom:25px;
#     box-shadow:0 5px 15px rgba(0,0,0,0.15);
# }

# .main-header img{
#     width:90px;
#     height:90px;
#     border-radius:12px;
# }

# .main-header h1{
#     margin:0;
#     font-size:28px;
# }

# .main-header p{
#     margin:5px 0;
#     font-size:15px;
#     opacity:0.95;
# }

# /* CARD */
# .custom-card{
#     background:white;
#     padding:20px;
#     border-radius:16px;
#     box-shadow:0 4px 12px rgba(0,0,0,0.08);
#     margin-bottom:15px;
# }

# /* INFO BOX */
# .info-box{
#     background:#eef8f0;
#     border-left:5px solid #2e7d32;
#     padding:15px;
#     border-radius:10px;
#     margin-top:12px;
# }

# /* IMAGE */
# img{
#     border-radius:12px;
# }

# </style>
# """, unsafe_allow_html=True)


# import streamlit as st

# # ================= CSS =================
# st.markdown("""
# <style>

# div[data-testid="stHorizontalBlock"]{
#     align-items:center;
# }

# .left-bar{
#     width:6px;
#     height:90px;
#     background:#2d6f90;
#     border-radius:20px;
#     margin:auto;
# }

# .school{
#     font-size:15px;
#     font-weight:700;
#     color:#444;
#     margin-bottom:2px;
# }

# .title{
#     font-size:34px;
#     font-weight:800;
#     color:#111;
#     line-height:1.2;
# }

# .subtitle{
#     font-size:16px;
#     color:#777;
# }

# </style>
# """, unsafe_allow_html=True)


# # ================= HEADER =================

# c1,c2,c3 = st.columns([0.15,0.7,7])

# with c1:
#     st.markdown(
#         '<div class="left-bar"></div>',
#         unsafe_allow_html=True
#     )

# with c2:
#     st.image("assets/logo.png", width=85)

# with c3:

#     st.markdown(
#     """
#     <div class="school">
#         TRƯỜNG ĐẠI HỌC THỦY LỢI
#     </div>

#     <div class="title">
#         Hệ thống phát hiện và phân loại sâu bệnh trên cây vải thiều
#     </div>

#     <div class="subtitle">
#         Ứng dụng mô hình CNN trong phân loại ảnh sâu bệnh cây trồng
#     </div>
#     """,
#     unsafe_allow_html=True
#     )

# # ======================
# # LOAD MODEL
# # ======================
# model = load_prediction_model()

# if model is None:
#     st.error("❌ Không tìm thấy model. Vui lòng train trước.")
#     st.stop()


# # ======================
# # UPLOAD SECTION
# # ======================
# st.markdown("""
# <div class="custom-card">
#     <h3>📤 Tải ảnh lá vải thiều</h3>
#     <p>Chọn ảnh để hệ thống phân tích và dự đoán bệnh.</p>
# </div>
# """, unsafe_allow_html=True)

# uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])


# # ======================
# # DISEASE INFO
# # ======================
# disease_info = {
#     "Anthrax Leaf": {
#         "symptom": "Đốm nâu/đen lan rộng gây hoại tử.",
#         "treatment": "Phun Copper, Mancozeb, Carbendazim; cắt lá bệnh."
#     },
#     "Bituminous Leaf": {
#         "symptom": "Lá sẫm màu do tiết nhựa bất thường.",
#         "treatment": "Diệt rệp, bọ trĩ; dùng Neem oil hoặc thuốc Imidacloprid."
#     },
#     "Curl Leaf": {
#         "symptom": "Lá xoăn, biến dạng.",
#         "treatment": "Phun Abamectin; bổ sung vi lượng."
#     },
#     "Deficiency Leaf": {
#         "symptom": "Lá vàng do thiếu dinh dưỡng.",
#         "treatment": "Bón NPK + Mg, Zn, Fe."
#     },
#     "Dry Leaf": {
#         "symptom": "Lá khô, cháy mép.",
#         "treatment": "Tưới nước hợp lý, bổ sung Kali + Canxi."
#     },
#     "Felt Leaf": {
#         "symptom": "Nhện lông nhung gây mảng bất thường.",
#         "treatment": "Phun Abamectin, dầu khoáng."
#     },
#     "Fungal Leaf Spot": {
#         "symptom": "Đốm nấm nâu/đen.",
#         "treatment": "Phun Mancozeb, Chlorothalonil."
#     },
#     "Healthy Leaf": {
#         "symptom": "Lá khỏe mạnh bình thường.",
#         "treatment": "Duy trì chăm sóc tốt."
#     },
#     "Leaf Blight": {
#         "symptom": "Cháy mép lá, vàng nâu.",
#         "treatment": "Phun Copper hydroxide; giảm đạm."
#     },
#     "Leaf Gall": {
#         "symptom": "Lá bị u, sần.",
#         "treatment": "Cắt bỏ lá bệnh, dùng thuốc trừ sâu."
#     }
# }


# # ======================
# # PREDICTION
# # ======================
# if uploaded_file is not None:

#     result = predict_uploaded_image(model, uploaded_file)

#     col1, col2 = st.columns(2)

#     with col1:
#         st.image(result["image"], caption="Ảnh đầu vào", use_container_width=True)

#     with col2:

#         confidence_percent = result["confidence"] * 100

#         st.markdown("## 📊 Kết quả")

#         if result["class_name"] == "Unknown":

#             st.error("❌ Không phải lá vải hoặc không chắc chắn")
#             st.metric("Độ tin cậy", f"{confidence_percent:.2f}%")

#         else:

#             st.success("✅ Dự đoán thành công")

#             st.write("**Tên lớp:**", result["class_name"])
#             st.write("**Độ tin cậy:**", f"{confidence_percent:.2f}%")

#             st.info(result["description"])

#             if result["class_name"] in disease_info:

#                 info = disease_info[result["class_name"]]

#                 st.markdown("### 🌿 Biểu hiện bệnh")
#                 st.write(info["symptom"])

#                 st.markdown("### 🛠️ Cách khắc phục")
#                 st.write(info["treatment"])


#     # ======================
#     # PROBABILITY TABLE
#     # ======================
#     st.markdown("---")
#     st.markdown("### 📈 Xác suất từng lớp")

#     df = pd.DataFrame({
#         "Lớp": result["class_names"],
#         "Xác suất (%)": [round(x * 100, 2) for x in result["all_predictions"]],
#     }).sort_values(by="Xác suất (%)", ascending=False)

#     st.dataframe(df, use_container_width=True)

# else:
#     st.info("📌 Vui lòng upload ảnh để bắt đầu")



