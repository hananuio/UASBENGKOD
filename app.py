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
gender_text = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
gender = 0 if gender_text == "Perempuan" else 1

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
