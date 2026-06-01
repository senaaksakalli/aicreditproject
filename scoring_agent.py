import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
import joblib

# 1. Temizlenmiş veriyi yükle
df = pd.read_csv('cleaned_credit_data.csv')

# 2. Kategorik verileri sayısal verilere çevir (DUMMIES)
df = pd.get_dummies(df)
target_column = 'loan_status'

# 3. NORMALİZASYON  [28]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)


joblib.dump(scaler, 'scaler.gz')
model_columns = df.drop(target_column, axis=1).columns.tolist()
joblib.dump(model_columns, 'model_columns.pkl')

# 4. SLIDING WINDOW  OLUŞTURMA [28], [29]
def create_sequences(data, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:(i + seq_length), :-1]) 
        y_seq.append(data[i + seq_length, -1])    
    return np.array(X_seq), np.array(y_seq)

pencere_boyutu = 12
X, y = create_sequences(scaled_data, pencere_boyutu)

# 5. Veriyi Ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. BiLSTM MODEL MİMARİSİ [9], [28]
print("\n--- BiLSTM Modeli Kuruluyor ---")
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # 1. Katman: Çift Yönlü LSTM
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    
    # 2. Katman: Çift Yönlü LSTM (Derin yapı)
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    
    # Karar Katmanları
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid') # 0-1 arası olasılık çıktısı
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Modeli Eğitme
print("\n--- BiLSTM Modeli Eğitiliyor (Epochs: 20) ---")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

model.save('bilstm_credit_model.h5')

# 9. Performans Analizi
y_probs = model.predict(X_test)
y_pred = (y_probs > 0.5).astype("int32")

print("\n--- BiLSTM PERFORMANS SONUÇLARI ---")
print(f"Accuracy: %{accuracy_score(y_test, y_pred)*100:.2f}")
print("MCC Skoru:", matthews_corrcoef(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))