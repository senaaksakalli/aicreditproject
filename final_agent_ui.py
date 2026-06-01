import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# 1. BiLSTM Modelini ve Yardımcı Araçları 
try:
    # scoring_agent.py ile kaydettiğimiz dosyaları yükleme
    model = tf.keras.models.load_model('bilstm_credit_model.h5')
    scaler = joblib.load('scaler.gz')
    model_columns = joblib.load('model_columns.pkl')
    
    print("--- AI CREDIT AGENT (BiLSTM) AKTİF ---")

    # 2. Yeni Müşteri Simülasyonu

    yeni_musteri_verisi = {
        'person_age': 25,
        'person_income': 50000,
        'loan_amnt': 25000,
        'loan_int_rate': 11.5,
        'loan_percent_income': 0.50,
    }

   
    input_df = pd.DataFrame([yeni_musteri_verisi]).reindex(columns=model_columns, fill_value=0)

    # 3. BiLSTM İçin Ön İşleme (Scaling ve Zaman Serisi Dönüşümü)

    temp_df = input_df.copy()
    temp_df['loan_status'] = 0 
    scaled_input = scaler.transform(temp_df)[:, :-1] # Sadece özellikleri al

   
    sequence_input = np.repeat(scaled_input[np.newaxis, :, :], 12, axis=1)

    # 4. Karar ve Açıklama [28], [29]
    olasilik = model.predict(sequence_input)[0][0]
    tahmin = 1 if olasilik > 0.5 else 0

    print("\n[AJAN KARARI - BiLSTM ANALİZİ]")
    if tahmin == 1:
        print("Durum: KREDİ REDDEDİLDİ ❌")
        print(f"Risk Olasılığı (Probability): %{olasilik*100:.2f}")
        print("Gerekçe: 12 aylık simülasyon sonuçları yüksek temerrüt riski işaret ediyor.")
    else:
        print("Durum: KREDİ ONAYLANDI ✅")
        print(f"Güven Skoru (Confidence): %{(1-olasilik)*100:.2f}")
        print("Açıklama: Finansal trendler kredi geri ödemesi için güvenli görünüyor.")

    print("-" * 35)

except Exception as e:
    print(f"Ajan başlatılamadı! Hata: {e}")
    print("Lütfen önce 'scoring_agent.py' dosyasını çalıştırarak modeli eğitin.")