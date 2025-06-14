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

# Input numerik
age = st.number_input("Umur (tahun)", 1, 100, value=25)
height = st.number_input("Tinggi Badan (meter)", 1.0, 2.5, value=1.70)
weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, value=70.0)
fcvc = st.slider("Frekuensi konsumsi sayuran (1–3)", 1.0, 3.0, value=2.0)
ncp = st.slider("Jumlah makan per hari", 1.0, 5.0, value=3.0)
ch2o = st.slider("Konsumsi air putih per hari (liter)", 1.0, 3.0, value=2.0)
faf = st.slider("Aktivitas fisik per minggu (jam)", 0.0, 20.0, value=2.0)
tue = st.slider("Waktu layar per hari (jam)", 0.0, 10.0, value=1.0)

# Konversi selectbox (string to int encoding)
gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
gender = 0 if gender == "Perempuan" else 1

calc_text = st.selectbox("Konsumsi Alkohol", ["Selalu", "Sering", "Kadang-kadang", "Tidak Pernah"])
calc = {"Selalu": 0, "Sering": 1, "Kadang-kadang": 2, "Tidak Pernah": 3}[calc_text]

favc = st.selectbox("Sering makan makanan berkalori tinggi?", ["Tidak", "Ya"])
favc = 0 if favc == "Tidak" else 1

scc = st.selectbox("Memilih makanan sehat?", ["Tidak", "Ya"])
scc = 0 if scc == "Tidak" else 1

smoke = st.selectbox("Merokok?", ["Tidak", "Ya"])
smoke = 0 if smoke == "Tidak" else 1

family_history = st.selectbox("Riwayat keluarga obesitas?", ["Tidak", "Ya"])
family_history = 0 if family_history == "Tidak" else 1

caec_text = st.selectbox("Ngemil/makan di luar jam makan:", ["Selalu", "Sering", "Kadang-kadang", "Tidak Pernah"])
caec = {"Selalu": 0, "Sering": 1, "Kadang-kadang": 2, "Tidak Pernah": 3}[caec_text]

mtrans_text = st.selectbox("Transportasi utama", ["Mobil", "Sepeda", "Motor", "Transportasi Umum", "Jalan Kaki"])
mtrans = {"Mobil": 0, "Sepeda": 1, "Motor": 2, "Transportasi Umum": 3, "Jalan Kaki": 4}[mtrans_text]

# Prediksi
if st.button("🔮 Prediksi"):
    try:
        # Susun input dan pastikan tipe data float
        sample = np.array([[age, gender, height, weight, calc, favc, fcvc, ncp,
                            scc, smoke, ch2o, family_history, faf, tue, caec, mtrans]],
                          dtype=np.float64)

        if sample.shape != (1, 16):
            st.error(f"Bentuk input salah: {sample.shape}, harus (1, 16)")
            st.stop()

        # Scaling dan prediksi
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)
        predicted_label = le_target.inverse_transform(prediction)

        st.success(f"✅ Prediksi Kategori Obesitas: **{predicted_label[0]}**")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi:\n\n{e}")
