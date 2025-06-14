import streamlit as st
import numpy as np
import joblib

# Load model, scaler, dan label encoder
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
le_target = joblib.load("label_encoder_target.pkl")

st.title("Prediksi Kategori Obesitas")

# Input form
age = st.number_input("Umur", min_value=1, max_value=100, value=25)
gender = st.selectbox("Jenis Kelamin", {"Female": 0, "Male": 1})
height = st.number_input("Tinggi (meter)", min_value=1.0, max_value=2.5, value=1.75)
weight = st.number_input("Berat (kg)", min_value=30.0, max_value=200.0, value=70.0)
calc = st.selectbox("Konsumsi Alkohol", {"Always": 0, "Frequently": 1, "Sometimes": 2, "no": 3})
favc = st.selectbox("Makan makanan berkalori tinggi?", {"no": 0, "yes": 1})
fcvc = st.slider("Frekuensi makan sayur (0-3)", 0.0, 3.0, 2.0)
ncp = st.slider("Jumlah makan per hari", 1.0, 5.0, 3.0)
scc = st.selectbox("Konsumsi makanan sehat?", {"no": 0, "yes": 1})
smoke = st.selectbox("Merokok?", {"no": 0, "yes": 1})
ch2o = st.slider("Konsumsi air (liter/hari)", 0.0, 3.0, 2.0)
family_history = st.selectbox("Riwayat keluarga obesitas?", {"no": 0, "yes": 1})
faf = st.slider("Aktivitas fisik (jam/minggu)", 0.0, 10.0, 2.0)
tue = st.slider("Waktu layar/hari (jam)", 0.0, 5.0, 1.0)
caec = st.selectbox("Makan di luar jam makan?", {"Always": 0, "Frequently": 1, "Sometimes": 2, "no": 3})
mtrans = st.selectbox("Transportasi utama", {
    "Automobile": 0,
    "Bike": 1,
    "Motorbike": 2,
    "Public_Transportation": 3,
    "Walking": 4
})

if st.button("Prediksi"):
    # Susun input
    sample = np.array([[age, gender, height, weight, calc, favc, fcvc, ncp,
                        scc, smoke, ch2o, family_history, faf, tue, caec, mtrans]])

    # Preprocessing
    sample_scaled = scaler.transform(sample)

    # Prediksi
    prediction = model.predict(sample_scaled)
    label = le_target.inverse_transform(prediction)

    st.success(f"Prediksi kategori: **{label[0]}**")
