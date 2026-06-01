import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. Veriyi Oku ve Temizle
try:
    df = pd.read_csv('credit_data.csv')
    df.ffill(inplace=True)
    
   
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 2. Ölçeklendirme 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(numeric_df)
    
    
    joblib.dump(scaler, 'scaler.gz')
    joblib.dump(numeric_df.columns.drop('loan_status').tolist(), 'model_columns.pkl')

    # 3. Sliding Window (Kayan Pencere) Oluşturma [28]
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length), :-1])
            y.append(data[i + seq_length, -1])
        return np.array(X), np.array(y)

    pencere_boyutu = 12
    X, y = create_sequences(scaled_data, pencere_boyutu)

    # 4. BiLSTM Model Mimarisi [29]
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 5. Modeli Eğit
    print("BiLSTM Modeli Eğitiliyor...")
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    
    # Modeli Kaydet
    model.save('bilstm_credit_model.h5')
    print("Model ve yardımcı dosyalar başarıyla kaydedildi!")

except Exception as e:
    print(f"Hata oluştu: {e}")