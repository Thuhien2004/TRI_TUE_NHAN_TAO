import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ===== CONFIG =====
st.set_page_config(
    page_title="Fashion AI",
    page_icon="👕",
    layout="wide"
)

# ===== STYLE =====
st.markdown("""
<div style="text-align:center;">
    <span style="font-size:120px;">👕</span><br>
    <span style="
        font-size:90px;
        font-weight:bold;
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;">
        Fashion AI
    </span>
</div>
""", unsafe_allow_html=True)

# ===== LOAD MODEL =====
model = tf.keras.models.load_model("model/fashion_model.h5", compile=False)
class_names = ['ao', 'ao_khoac', 'giay', 'phu_kien', 'quan', 'tui', 'vay']

# ===== PREDICT FUNCTION =====
def predict_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return model.predict(img)

# ===== SUGGEST FUNCTION =====
def suggest_outfit(label, weather, occasion, style):
    outfit = []

    if label == "ao":
        outfit += ["Quần jean", "Giày sneaker"]
    elif label == "vay":
        outfit += ["Giày cao gót", "Túi xách"]

    if weather == "Lạnh":
        outfit.append("Áo khoác")
    elif weather == "Nóng":
        outfit.append("Áo mỏng")

    if occasion == "Đi học":
        outfit.append("Ba lô")
    elif occasion == "Đi chơi":
        outfit.append("Túi thời trang")

    if style == "Trẻ trung":
        outfit.append("Phong cách năng động")
    else:
        outfit.append("Phong cách thanh lịch")

    return list(set(outfit))


# ===== LAYOUT =====
col1, col2 = st.columns(2)

# ===== LEFT =====
with col1:
    st.subheader("📸 Upload ảnh")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        with st.spinner("🔍 Đang phân tích..."):
            time.sleep(1)
            pred = predict_image(image)

        class_idx = np.argmax(pred)
        confidence = np.max(pred)

        # 🔥 LƯU VÀO SESSION
        st.session_state.label = class_names[class_idx]

        st.success(f"👉 {st.session_state.label} ({confidence:.2f})")
        st.progress(float(confidence))


# ===== RIGHT =====
with col2:
    st.subheader("🎯 Gợi ý trang phục")

    weather = st.selectbox("🌤️ Thời tiết", ["Nóng", "Lạnh"])
    occasion = st.selectbox("📍 Hoàn cảnh", ["Đi học", "Đi chơi"])
    style = st.selectbox("✨ Phong cách", ["Trẻ trung", "Thanh lịch"])

    if st.button("✨ Gợi ý ngay"):
        if "label" in st.session_state:
            outfit = suggest_outfit(
                st.session_state.label,
                weather,
                occasion,
                style
            )

            st.success("🔥 Outfit đề xuất:")
            for item in outfit:
                st.markdown(f"👉 **{item}**")
        else:
            st.warning("⚠️ Hãy upload ảnh trước!")