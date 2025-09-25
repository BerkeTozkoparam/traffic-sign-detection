import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ========================
# SAYFA AYARI
# ========================
st.set_page_config(page_title="YOLOv8 Tabela Tespiti", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚦 YOLOv8 Tabela Tespiti Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

# ========================
# MODEL YÜKLEME
# ========================
@st.cache_resource
def load_model(path):
    return YOLO(path)

model_path = "/Users/berkebarantozkoparan/Desktop/car/best.pt"  # model yolunu ayarla
model = load_model(model_path)

# ========================
# SIDEBAR AYARLAR
# ========================
st.sidebar.header("⚙️ Ayarlar")
confidence = st.sidebar.slider("Minimum Güven Skoru", 0.0, 1.0, 0.25, 0.05)
mode = st.sidebar.radio("Veri Kaynağı", ["📂 Görsel", "📷 Webcam"])
download_format = st.sidebar.radio("Sonuç İndirme Formatı", ["CSV", "JSON"])

# ========================
# DETAY TABLOSU FONKSİYONU
# ========================
def results_to_df(results, model):
    data = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()
        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = box.astype(int)
            data.append({
                "Sınıf": model.names[int(cls)],
                "Güven": round(float(conf), 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
    return pd.DataFrame(data)

# ========================
# GRAFİK ÇİZİM
# ========================
def plot_class_distribution(df):
    if df.empty:
        return
    fig, ax = plt.subplots()
    df["Sınıf"].value_counts().plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
    ax.set_title("Tespit Edilen Tabela Dağılımı")
    ax.set_xlabel("Tabela Sınıfı")
    ax.set_ylabel("Adet")
    st.pyplot(fig)

# ========================
# GÖRSEL MODU
# ========================
if mode == "📂 Görsel":
    uploaded_file = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        results = model(img, conf=confidence)
        annotated_img = results[0].plot()

        # 3 sütunlu düzen
        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col1:
            st.subheader("📥 Yüklenen Görsel")
            st.image(img, channels="BGR")
        with col2:
            st.subheader("📌 Tespit Sonucu")
            st.image(annotated_img, channels="BGR")
        with col3:
            st.subheader("📊 Detaylar")
            df = results_to_df(results, model)
            st.dataframe(df, use_container_width=True)

            # İndirme
            if not df.empty:
                if download_format == "CSV":
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ CSV indir", csv, "results.csv", "text/csv")
                else:
                    json_data = df.to_json(orient="records").encode("utf-8")
                    st.download_button("⬇️ JSON indir", json_data, "results.json", "application/json")

            # İstatistik grafiği
            st.subheader("📈 İstatistikler")
            plot_class_distribution(df)

            # DİKKAT uyarısı
            if "Dur" in df["Sınıf"].values:
                st.error("⚠️ DİKKAT! 'Dur' tabelası tespit edildi.")

# ========================
# WEBCAM MODU
# ========================
elif mode == "📷 Webcam":
    st.info("Canlı yayın için butona basınız.")
    run = st.checkbox("▶️ Webcam'i Başlat")
    stframe = st.empty()
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Kamera bulunamadı")
                break

            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()
            df = results_to_df(results, model)
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

            if not df.empty and "Dur" in df["Sınıf"].values:
                st.error("⚠️ DİKKAT! 'Dur' tabelası tespit edildi.")

        cap.release()
