import streamlit as st
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# ======================================================================================
# 1. Konfigurasi Halaman
# ======================================================================================
st.set_page_config(page_title="Rekomendasi Musik Emosi", layout="wide")

# ======================================================================================
# 2. Inisialisasi Session State (Untuk Navigasi & Data)
# ======================================================================================
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'  # Default halaman awal

if 'last_emotion' not in st.session_state:
    st.session_state['last_emotion'] = None

if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = pd.DataFrame()

# Fungsi untuk pindah halaman
def navigate_to(page_name):
    st.session_state['page'] = page_name

# ======================================================================================
# 3. Pemuatan Aset (Model dan Data)
# ======================================================================================
@st.cache_resource
def load_assets():
    try:
        # Dapatkan lokasi absolut folder tempat app.py berada
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Gabungkan folder tersebut dengan nama file
        model_path = os.path.join(base_dir, 'final_model_lite.h5')
        cascade_path = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')

        # Load Model
        if not os.path.exists(model_path):
            st.error(f"File tidak ditemukan: {model_path}")
            return None, None
            
        model = load_model(model_path)
        
        # Load Cascade (Cek file lokal dulu, kalau ga ada pakai bawaan cv2)
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            # Fallback ke library bawaan jika file xml tidak terupload
            st.warning("Menggunakan haarcascade bawaan library (file xml lokal tidak ditemukan).")
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
        return model, face_cascade
    except Exception as e:
        st.error(f"Error saat memuat aset: {e}")
        return None, None

@st.cache_data
def load_music_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # PASTIKAN NAMA FILE SAMA PERSIS DENGAN YANG DI GITHUB (data_moods.xlsx atau data_moods3.xlsx?)
        excel_path = os.path.join(base_dir, 'data_moods3.xlsx') 
        
        if not os.path.exists(excel_path):
            st.error(f"File excel tidak ditemukan di: {excel_path}")
            return pd.DataFrame()
            
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        st.error(f"Gagal membaca data musik: {e}")
        return pd.DataFrame()

# Inisialisasi Aset
EMOTION_CLASSES = ['Happy', 'Neutral', 'Sad'] 
IMG_SIZE = (224, 224) 
model, face_cascade = load_assets()
df_music = load_music_data()

# ======================================================================================
# 4. Fungsi-Fungsi Logika (Deteksi & API)
# ======================================================================================

def detect_emotion_from_image(image):
    if model is None or face_cascade is None:
        return image, "Error: Model tidak termuat"

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return image, "Wajah tidak terdeteksi"

    (x, y, w, h) = faces[0]
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    face_roi = gray_image[y:y+h, x:x+w]
    
    try:
        roi_resized = cv2.resize(face_roi, IMG_SIZE)
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2RGB)
        roi_normalized = roi_rgb / 255.0
        roi_expanded = np.expand_dims(roi_normalized, axis=0)

        prediction = model.predict(roi_expanded)
        predicted_emotion = EMOTION_CLASSES[np.argmax(prediction)]
        return image, predicted_emotion
    except Exception as e:
        return image, f"Error Processing: {e}"

def get_new_recommendations(emotion, num_recommendations=10):
    if df_music.empty:
        return
    recommended_songs = df_music[df_music['mood'].str.lower() == emotion.lower()]
    if not recommended_songs.empty:
        st.session_state['recommendations'] = recommended_songs.sample(n=min(num_recommendations, len(recommended_songs)))
    else:
        st.session_state['recommendations'] = pd.DataFrame()

def save_feedback_to_google_sheets(nama, kepuasan, saran):
    """
    Menyimpan data feedback ke Google Sheets.
    PENTING: Anda harus mengatur 'st.secrets' di Streamlit Cloud atau file .streamlit/secrets.toml
    """
    try:
        # Mengambil kredensial dari st.secrets (aman untuk deploy)
        # Struktur secrets.toml harus:
        # [gcp_service_account]
        # type = "service_account"
        # ... (isi json lengkap) ...
        
        # SEMENTARA: Jika belum setup secrets, kita skip bagian ini agar app tidak crash
        if "gcp_service_account" not in st.secrets:
            st.warning("Google Sheets API belum dikonfigurasi di st.secrets. Feedback hanya simulasi.")
            return True

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
        client = gspread.authorize(creds)
        
        # Ganti dengan Nama Spreadsheet Anda (Harus sudah dishare ke email service account)
        sheet = client.open("Feedback_App_Musik").sheet1 
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Menambahkan baris baru: [Waktu, Nama, Rating, Saran]
        sheet.append_row([timestamp, nama, kepuasan, saran])
        return True
        
    except Exception as e:
        st.error(f"Gagal menyimpan ke Google Sheets: {e}")
        return False

# ======================================================================================
# 5. Sidebar (Navigasi Tombol)
# ======================================================================================
with st.sidebar:
    st.title("Navigasi")
    
    # Menggunakan callback args untuk mengubah state tanpa rerun yang mengacaukan
    st.button("🏠 Home", on_click=navigate_to, args=('home',), use_container_width=True)
    st.button("📸 Deteksi Webcam", on_click=navigate_to, args=('webcam',), use_container_width=True)
    st.button("🖼️ Upload Gambar", on_click=navigate_to, args=('upload',), use_container_width=True)
    st.button("📝 Beri Feedback", on_click=navigate_to, args=('feedback',), use_container_width=True)

    st.divider()
    st.info("Navigasi menggunakan tombol Dashboard.")

# ======================================================================================
# 6. Logika Halaman Utama
# ======================================================================================

# --- HALAMAN HOME ---
# --- HALAMAN HOME ---
if st.session_state['page'] == 'home':
    # 1. Judul & Sambutan Hangat
    st.title("👋 Selamat Datang di MoodMelody!")
    st.subheader("Temukan *Soundtrack* Hidupmu Berdasarkan Emosi Wajah")
    
    st.markdown("---")
    
    # 2. PENGUMUMAN PENTING (Spotify Login)
    # Menggunakan st.info atau st.warning agar mencolok
    st.info("""
    🔔 **PENTING: Persiapan Sebelum Memulai**
    
    Agar lagu dapat diputar secara langsung dan lancar, pastikan Anda **sudah Login ke akun Spotify** di browser ini (Chrome/Edge/dll) sebelum menggunakan fitur deteksi.
    
    👉 [Klik di sini untuk Login Spotify Web Player](https://open.spotify.com) (buka di tab baru)
    """)

    st.markdown("### 🚀 Jelajahi Fitur Kami")
    st.write("Aplikasi ini menggunakan Kecerdasan Buatan (AI) untuk membaca ekspresi wajahmu dan menyajikan musik yang paling *relate* dengan perasaanmu saat ini.")

    # 3. PENJELASAN FITUR (Menggunakan Columns agar layout lebih menarik)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📸 Deteksi Webcam")
        st.write("""
        **Cerminkan Perasaanmu Secara Langsung!**
        
        Fitur ini akan memindai wajahmu secara *real-time*. 
        * **Syarat:** Mohon **izinkan akses kamera** pada browser Anda saat diminta.
        * **Cara:** Klik tombol, senyum (atau cemberut), dan biarkan AI kami memilihkan lagu hits yang pas buat mood kamu sekarang!
        """)
        if st.button("Coba Webcam ➡️"):
            navigate_to('webcam')
            st.rerun()

    with col2:
        st.markdown("### 🖼️ Upload Gambar")
        st.write("""
        **Punya Foto Candid yang Estetik?**
        
        Tidak ingin menyalakan kamera? Tenang saja!
        * **Cara:** Unggah foto selfie atau potret wajah dari galerimu.
        * **Sistem:** Kami akan menganalisis file gambar tersebut dan memberikan rekomendasi musik yang sesuai dengan ekspresi di foto.
        """)
        if st.button("Upload Foto ➡️"):
            navigate_to('upload')
            st.rerun()

    with col3:
        st.markdown("### 📝 Beri Feedback")
        st.write("""
        **Suaramu Sangat Berarti!**
        
        Selesai mendengarkan musik?
        * **Harapan Kami:** Bantu kami mengembangkan sistem ini menjadi lebih cerdas dengan mengisi ulasan singkat.
        * **Isi:** Berikan rating kepuasan dan saran fitur yang kamu inginkan. Masukanmu adalah semangat kami!
        """)
        if st.button("Isi Feedback ➡️"):
            navigate_to('feedback')
            st.rerun()

    st.markdown("---")
    
    # 4. Footer / Quote Pemanis
    st.markdown("""
    <div style='text-align: center; color: grey;'>
        <i>"Where words fail, music speaks." — Hans Christian Andersen</i><br>
        Dibuat dengan ❤️ untuk Skripsi/Tugas Akhir
    </div>
    """, unsafe_allow_html=True)

# --- HALAMAN DETEKSI (WEBCAM & UPLOAD GABUNGAN LOGIKA) ---
elif st.session_state['page'] in ['webcam', 'upload']:
    mode = "Webcam" if st.session_state['page'] == 'webcam' else "Upload"
    st.title(f"Mode: {mode}")

    image_buffer = None
    if st.session_state['page'] == 'webcam':
        image_buffer = st.camera_input("Ambil Gambar", label_visibility="visible")
    else:
        image_buffer = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    # Proses Gambar
    if image_buffer:
        # Konversi ke OpenCV format
        if st.session_state['page'] == 'webcam':
            bytes_data = image_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            file_bytes = np.asarray(bytearray(image_buffer.read()), dtype=np.uint8)
            cv2_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.session_state['processed_image'] = cv2_img.copy()
        
        # Deteksi
        with st.spinner('Menganalisis ekspresi wajah...'):
            _, detected_emotion = detect_emotion_from_image(st.session_state['processed_image'])
        
        st.session_state['last_emotion'] = detected_emotion
        
        # Rekomendasi
        if detected_emotion not in ["Wajah tidak terdeteksi", "Error: Model tidak termuat"]:
            get_new_recommendations(detected_emotion)
        else:
            st.session_state['recommendations'] = pd.DataFrame()

    # Tampilan Hasil (Sama untuk kedua mode)
    if st.session_state.get('last_emotion'):
        st.divider()
        detected_emotion = st.session_state['last_emotion']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if 'processed_image' in st.session_state:
                # Convert BGR to RGB for Streamlit display
                rgb_img = cv2.cvtColor(st.session_state['processed_image'], cv2.COLOR_BGR2RGB)
                st.image(rgb_img, caption="Analisis Wajah", use_container_width=True)

        with col2:
            if detected_emotion in ["Wajah tidak terdeteksi", "Error: Model tidak termuat"]:
                st.warning(f"⚠️ {detected_emotion}")
            else:
                st.success(f"### Emosi: {detected_emotion}")
                
                if st.button("Acak Lagu Lain 🔄", key="shuffle_btn"):
                    get_new_recommendations(detected_emotion)
                    st.rerun()

                if not st.session_state['recommendations'].empty:
                    st.write("### 🎧 Rekomendasi Musik:")
                    for _, row in st.session_state['recommendations'].iterrows():
                        with st.expander(f"**{row['name']}** - {row['artist']}"):
                             # Asumsi kolom 'id' adalah Spotify Track ID
                            spotify_url = f"https://open.spotify.com/embed/track/{row['id']}?utm_source=generator"
                            st.markdown(f'<iframe src="{spotify_url}" width="100%" height="80" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>', unsafe_allow_html=True)
                else:
                    st.info("Belum ada data musik untuk emosi ini.")

# --- HALAMAN FEEDBACK ---
elif st.session_state['page'] == 'feedback':
    st.title("📝 Feedback Pengguna")
    
    # 1. Cek dulu, apakah di sesi ini dia sudah pernah submit?
    if 'sudah_isi_feedback' not in st.session_state:
        st.session_state['sudah_isi_feedback'] = False

    # 2. Logika Tampilan
    if st.session_state['sudah_isi_feedback']:
        # Tampilan JIKA SUDAH mengisi
        st.success("✅ Terima kasih! Anda sudah mengirimkan masukan untuk sesi ini.")
        st.info("Jika Anda ingin mengisi ulang, silakan refresh halaman browser Anda.")
        
        # Opsi kembali ke home
        if st.button("Kembali ke Home"):
            navigate_to('home')
            st.rerun()
            
    else:
        # Tampilan JIKA BELUM mengisi (Formulir Asli)
        st.write("Bantu kami meningkatkan aplikasi ini dengan memberikan ulasan Anda.")

        with st.form("form_feedback"):
            nama = st.text_input("Nama Anda (Wajib)")
            
            st.write("Tingkat Kepuasan (Wajib)")
            sentiment = st.feedback("stars") 
            
            saran = st.text_area("Saran & Masukan (Opsional)")
            
            # Disable tombol jika sedang proses (opsional, tapi bagus untuk UX)
            submit_btn = st.form_submit_button("Kirim Feedback")

            if submit_btn:
                if not nama:
                    st.error("Mohon isi Nama Anda.")
                elif sentiment is None:
                    st.error("Mohon berikan rating kepuasan.")
                else:
                    rating_scale = sentiment + 1 
                    
                    with st.spinner("Mengirim data..."):
                        success = save_feedback_to_google_sheets(nama, rating_scale, saran)
                        
                    if success:
                        # 3. KUNCI RAHASIANYA DI SINI
                        # Ubah status session state menjadi True
                        st.session_state['sudah_isi_feedback'] = True
                        
                        st.balloons()
                        # Rerun agar tampilan langsung berubah ke pesan "Terima Kasih"

                        st.rerun()
