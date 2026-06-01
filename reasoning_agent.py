import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
import shap # BiLSTM analizi için 

# 1. Model ve Verileri Yükleme
model = tf.keras.models.load_model('bilstm_credit_model.h5')
scaler = joblib.load('scaler.gz')
model_columns = joblib.load('model_columns.pkl')

# Analiz için örnek bir veri kümesi 
df = pd.read_csv('cleaned_credit_data.csv').head(100)
df_dummies = pd.get_dummies(df).reindex(columns=model_columns + ['loan_status'], fill_value=0)
X_sample = scaler.transform(df_dummies)[:, :-1]

# BiLSTM 3D veri beklediği için (12 aylık pencere simülasyonu)
X_sample_3d = np.repeat(X_sample[:, np.newaxis, :], 12, axis=1)

# 2. SHAP ile BiLSTM Analizi (Gerekçelendirme Mekanizması) [28], [29]
explainer = shap.GradientExplainer(model, X_sample_3d)
shap_values = explainer.shap_values(X_sample_3d)

# Zaman boyutundaki (12 ay) SHAP değerlerinin ortalaması
avg_shap_values = np.mean(np.abs(shap_values[0]), axis=1) 
feature_importance = np.mean(avg_shap_values, axis=0)

# 3. Görselleştirme
importances = pd.DataFrame({
    'Ozellik': model_columns,
    'Onem': feature_importance
}).sort_values(by='Onem', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Onem', y='Ozellik', data=importances.head(10), palette='viridis')
plt.title('BiLSTM Modelinin Kararını Etkileyen En Kritik 10 Faktör (SHAP Analizi)')
plt.xlabel('Etki Şiddeti (SHAP Value)')
plt.show()

print("\n--- BiLSTM Ajan Analizi ---")
top_feature = importances.iloc[0]['Ozellik']
print(f"Ajan tespiti: BiLSTM modeline göre en kritik faktör: {top_feature}")
print("Analiz Yöntemi: SHAP (SHapley Additive exPlanations) [28]")