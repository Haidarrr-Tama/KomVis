import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Deteksi Real-time dengan InceptionV3",
    page_icon="📸",
    layout="centered"
)

st.title("Deteksi Gambar Real-time Menggunakan InceptionV3")
st.write("Aplikasi ini mendeteksi kelas dari gambar yang diambil dari kamera secara real-time.")

# --- Bagian 1: Memuat Model ---
@st.cache_resource
def load_model():
    try:
        model_path = 'model_inceptionv3_best.h5'
        gdrive_file_id = '1zcJCu0HcG_av-q1pPuA-KNmAfh3gruDz'  # Ganti dengan file ID kamu
        gdrive_url = f'https://drive.google.com/uc?id={gdrive_file_id}'

        # Cek apakah file model sudah ada
        if not os.path.exists(model_path):
            st.info("Mengunduh model dari Google Drive...")
            gdown.download(gdrive_url, model_path, quiet=False)

        # Load model setelah diunduh
        model = tf.keras.models.load_model(model_path)
        st.success("Model InceptionV3 berhasil dimuat!")
        return model

    except Exception as e:
        st.error(f"Gagal memuat model. Error: {e}")
    return None

model = load_model()

# --- Bagian 2: Definisi Kelas ---
# Ganti dengan daftar kelas yang sesuai dengan urutan output model Anda
# Contoh: Jika model Anda mengklasifikasikan 'COVID', 'Normal', 'Pneumonia'
CLASS_NAMES = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'healthy', 'powdery_mildew'] # Sesuaikan dengan kelas Anda!
INPUT_SHAPE = (299, 299) # InceptionV3 biasanya menerima input 224x224

# --- Bagian 3: Fungsi Prediksi ---
def preprocess_image(image_array, target_size):
    # Mengubah array numpy ke PIL Image untuk resize kualitas lebih baik jika diperlukan
    image_pil = Image.fromarray(image_array)
    image_resized = image_pil.resize(target_size)
    image_array_resized = img_to_array(image_resized)
    image_array_normalized = image_array_resized / 255.0  # Normalisasi
    image_array_expanded = np.expand_dims(image_array_normalized, axis=0) # Tambah batch dimension
    return image_array_expanded

def predict_frame(frame, model, class_names, input_shape):
    # Preprocessing frame
    preprocessed_frame = preprocess_image(frame, input_shape)

    # Prediksi
    predictions = model.predict(preprocessed_frame)
    
    # Dapatkan indeks kelas dengan probabilitas tertinggi
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions[0]) * 100

    return predicted_class_name, confidence

# --- Bagian 4: Stream Video dari Kamera ---
if model:
    st.subheader("Ambil Gambar dari Kamera")
    st.write("Klik 'Mulai Deteksi' untuk mengaktifkan kamera dan melihat hasil deteksi real-time.")

    # Placeholder untuk menampilkan video dan teks deteksi
    frame_placeholder = st.empty()
    detection_text_placeholder = st.empty()

    start_button = st.button("Mulai Deteksi")
    stop_button = st.button("Hentikan Deteksi")

    if start_button:
        st.session_state.run_camera = True
    if stop_button:
        st.session_state.run_camera = False

    if 'run_camera' not in st.session_state:
        st.session_state.run_camera = False

    if st.session_state.run_camera:
        # Menggunakan komponen `st.camera_input` untuk akses kamera.
        # Catatan: `st.camera_input` mengambil foto satu per satu,
        # untuk video stream real-time yang sebenarnya Anda perlu menggunakan
        # library eksternal seperti `webrtc_streamlit` atau pendekatan berbasis JS.
        # Namun, untuk demo cepat dan sederhana, ini cukup.
        
        # Untuk real-time streaming, kita perlu pendekatan yang berbeda:
        # Menggunakan `webrtc_streamlit` untuk live stream
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
        
        class VideoProcessor(VideoTransformerBase):
            def __init__(self):
                self.model = model
                self.class_names = CLASS_NAMES
                self.input_shape = INPUT_SHAPE
                self.result_text = "Menunggu deteksi..."

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24") # Mengubah frame ke numpy array (BGR)
                
                # Konversi BGR ke RGB karena InceptionV3 di Keras/TF biasanya expects RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Lakukan prediksi
                predicted_class_name, confidence = predict_frame(img_rgb, self.model, self.class_names, self.input_shape)
                self.result_text = f"Deteksi: {predicted_class_name} ({confidence:.2f}%)"

                # Tulis hasil di frame (Opsional: untuk feedback visual langsung di video)
                cv2.putText(img, self.result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                return img # Mengembalikan frame yang sudah diolah

        st.info("Memulai stream kamera. Harap izinkan akses kamera di browser Anda.")
        
        ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            video_transformer_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True, # Penting untuk performa
        )

        if ctx.state.playing:
            st.write("Stream kamera aktif. Melihat hasil deteksi...")
            # Menampilkan hasil deteksi di bawah video (bukan di dalam frame)
            # Ini akan diupdate oleh VideoProcessor jika kita memodifikasi untuk mengirim balik
            # informasi ke Streamlit main thread, atau kita bisa mengakses ctx.video_transformer.result_text
            # secara langsung jika Streamlit memungkinkan (biasanya tidak secara langsung seperti ini).
            # Untuk demo, kita asumsikan output visual di dalam frame cukup.
            # Jika Anda ingin teks terpisah, Anda perlu mekanisme komunikasi thread.
            
            # Sebagai work-around sederhana untuk menampilkan teks terpisah:
            if ctx.video_transformer:
                detection_text_placeholder.write(ctx.video_transformer.result_text)
                # Anda bisa menambahkan loop untuk terus memperbarui ini, tapi webrtc_streamer
                # sudah menangani streaming, jadi ini mungkin tidak ideal untuk pembaruan sangat cepat.
                # Untuk pembaruan teks secara cepat, lebih baik tampilkan langsung di overlay canvas
                # atau di dalam frame seperti yang sudah dilakukan di `transform` method.

    else:
        st.warning("Tekan 'Mulai Deteksi' untuk memulai.")

st.markdown("---")
st.write("Aplikasi ini dibuat untuk tujuan demonstrasi deteksi objek real-time dengan model InceptionV3.")