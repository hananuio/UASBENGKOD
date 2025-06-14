import streamlit as st
import numpy as np
import joblib

# Load model, scaler, dan label encoder
try:
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_target = joblib.load("label_encoder_target.pkl")
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

st.title("🔍 Prediksi Kategori Obesitas Berdasarkan Gaya Hidup")
st.markdown("Masukkan informasi pribadi dan kebiasaan Anda di bawah ini:")

# Input fitur
age = st.number_input("Umur (tahun)", 1, 100, value=25)
gender = st.selectbox("Jenis Kelamin", {"Perempuan": 0, "Laki-laki": 1})
height = st.number_input("Tinggi Badan (meter)", 1.0, 2.5, value=1.70)
weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, value=70.0)
calc = st.selectbox("Konsumsi Alkohol", {"Selalu": 0, "Sering": 1, "Kadang-kadang": 2, "Tidak Pernah": 3})
favc = st.selectbox("Sering makan makanan berkalori tinggi?", {"Tidak": 0, "Ya": 1})
fcvc = st.slider("Frekuensi konsumsi sayuran (1–3)", 1.0, 3.0, value=2.0)
ncp = st.slider("Jumlah makan per hari", 1.0, 5.0, value=3.0)
scc = st.selectbox("Memilih makanan sehat?", {"Tidak": 0, "Ya": 1})
smoke = st.selectbox("Merokok?", {"Tidak": 0, "Ya": 1})
ch2o = st.slider("Konsumsi air putih per hari (liter)", 1.0, 3.0, value=2.0)
family_history = st.selectbox("Riwayat keluarga obesitas?", {"Tidak": 0, "Ya": 1})
faf = st.slider("Aktivitas fisik per minggu (jam)", 0.0, 20.0, value=2.0)
tue = st.slider("Waktu layar per hari (jam)", 0.0, 10.0, value=1.0)
caec = st.selectbox("Ngemil/makan di luar jam makan:", {
    "Selalu": 0, "Sering": 1, "Kadang-kadang": 2, "Tidak Pernah": 3
})
mtrans = st.selectbox("Transportasi utama", {
    "Mobil": 0, "Sepeda": 1, "Motor": 2, "Transportasi Umum": 3, "Jalan Kaki": 4
})

# Prediksi
if st.button("🔮 Prediksi"):
    try:
        # Buat array input sesuai urutan training
        sample = np.array([[age, gender, height, weight, calc, favc, fcvc, ncp,
                            scc, smoke, ch2o, family_history, faf, tue, caec, mtrans]],
                          dtype=np.float64)

        # Validasi bentuk array
        if sample.shape != (1, 16):
            st.error(f"Input tidak valid. Diharapkan shape (1, 16), ditemukan {sample.shape}")
            st.stop()

        # Transformasi dan prediksi
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)
        predicted_label = le_target.inverse_transform(prediction)

        st.success(f"✅ Prediksi Kategori Obesitas: **{predicted_label[0]}**")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi:\n\n{e}")
