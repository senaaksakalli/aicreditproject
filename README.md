# AI CREDIT PROJECT
# Finansal Risk Yönetiminde BiLSTM Tabanlı Kredi Skorlama ve Temerrüt Tahmini

> Çift Yönlü LSTM mimarisine dayanan, uçtan uca otonom çalışan dört modüllü bir yapay zekâ ajan sistemi.

---

## Proje Hakkında

Bu proje, finansal risk yönetimi alanında kredi başvurularının temerrüt riskini tahmin etmeye yönelik bir yapay zekâ sistemi sunmaktadır. Geleneksel makine öğrenmesi yöntemlerinin zamansal bağımlılıkları modellemedeki kısıtlarını aşmak amacıyla BiLSTM (Bidirectional Long Short-Term Memory) mimarisi ve Kayan Pencere (Sliding Window) tekniği bir arada kullanılmıştır.

Sistem; veri işleme, model eğitimi, açıklanabilirlik analizi ve karar üretme aşamalarını insan müdahalesi gerektirmeksizin yöneten dört bağımsız Python modülünden oluşmaktadır.

---

## Sistem Mimarisi

```
Ham Veri (credit_data.csv)
        │
        ▼
┌─────────────────────────────────┐
│  Modül 1 — data_agent.py        │
│  ffill · MinMaxScaler · one-hot │
│  Sliding Window (n=12)          │
│  Çıktı: Tensör (32.569, 12, 26) │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Modül 2 — scoring_agent.py     │
│  BiLSTM Eğitimi (20 epoch)      │
│  Adam · Binary Cross-Entropy    │
│  Çıktı: bilstm_credit_model.h5  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Modül 3 — reasoning_agent.py   │
│  SHAP GradientExplainer         │
│  Özellik önem sıralaması        │
│  Çıktı: SHAP görselleri         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Modül 4 — final_agent_ui.py    │
│  Yeni müşteri değerlendirmesi   │
│  P > 0.5 → RED / P ≤ 0.5 → ONAY│
│  Çıktı: Risk raporu             │
└─────────────────────────────────┘
```

---

## Veri Seti

| Özellik | Değer |
|---|---|
| Kaynak | Kaggle (açık kaynak) |
| Kayıt sayısı | 32.581 |
| Özellik sayısı | 12 (işlem sonrası 26) |
| Hedef değişken | `loan_status` (0: Geri ödendi / 1: Temerrüt) |
| Sınıf dağılımı | %78.2 geri ödendi · %21.8 temerrüt |

---

## Model Mimarisi (BiLSTM)

```
Input        →  (batch, 12, 26)
BiLSTM-1     →  64 birim × 2 = 128  |  return_sequences=True
Dropout      →  0.2
BiLSTM-2     →  32 birim × 2 = 64   |  return_sequences=False
Dropout      →  0.2
Dense        →  16 nöron  |  ReLU
Output       →  1 nöron   |  Sigmoid  →  P(temerrüt) ∈ [0,1]

Toplam parametre  :  88.865
Optimize edici    :  Adam (lr=0.001)
Kayıp fonksiyonu  :  Binary Cross-Entropy
Eğitim            :  20 epoch · batch=32
```

---

## Performans Sonuçları

| Model | Accuracy | F1-Score | MCC | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Lojistik Reg. | 0.814 | 0.647 | 0.540 | 0.871 | 0.720 |
| Random Forest | 0.933 | 0.824 | 0.798 | 0.930 | 0.882 |
| **XGBoost** | 0.922 | 0.827 | 0.782 | **0.948** | **0.906** |
| **LightGBM** | 0.925 | 0.825 | 0.779 | **0.950** | **0.908** |
| MLP | 0.920 | 0.789 | 0.754 | 0.919 | 0.859 |
| BiLSTM (mevcut) | 0.822 | 0.077 | 0.135 | 0.665 | 0.321 |

> **Not:** Mevcut BiLSTM modeli temel konfigürasyonla eğitilmiştir. SMOTE entegrasyonu ve uzun eğitim süresiyle performansın karşılaştırma modelleriyle rekabet edebilir düzeye ulaşması beklenmektedir.

---

## Kurulum

```bash
# Repoyu klonla
git clone https://github.com/kullanici-adi/bilstm-kredi-skorlama.git
cd bilstm-kredi-skorlama

# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## Kullanım

Modüller sırasıyla çalıştırılmalıdır:

```bash
# 1. Veriyi işle ve sekansları oluştur
python data_agent.py

# 2. BiLSTM modelini eğit
python scoring_agent.py

# 3. SHAP açıklanabilirlik analizi
python reasoning_agent.py

# 4. Yeni müşteri değerlendirmesi
python final_agent_ui.py
```

---

## Gereksinimler

```
tensorflow>=2.10
scikit-learn>=1.1
pandas>=1.5
numpy>=1.23
xgboost>=1.7
lightgbm>=3.3
shap>=0.41
joblib>=1.2
matplotlib>=3.6
seaborn>=0.12
imbalanced-learn>=0.10
```

---

## Proje Yapısı

```
bilstm-kredi-skorlama/
│
├── data_agent.py           # Veri işleme ve sekans oluşturma
├── scoring_agent.py        # BiLSTM model eğitimi
├── reasoning_agent.py      # SHAP açıklanabilirlik analizi
├── final_agent_ui.py       # Karar ajanı arayüzü
├── app_agent.py            # Streamlit arayüzü
│
├── credit_data.csv         # Ham veri seti
├── cleaned_credit_data.csv # Temizlenmiş veri seti
│
├── bilstm_credit_model.h5  # Eğitilmiş BiLSTM modeli
├── scaler.gz               # MinMaxScaler nesnesi
├── model_columns.pkl       # Özellik sütun isimleri
│
├── outputs/                # Grafikler ve analiz çıktıları
│   ├── grafik_1_confusion_matrix.png
│   ├── grafik_2_roc_egrisi.png
│   ├── grafik_3_pr_egrisi.png
│   ├── grafik_4_model_karsilastirma.png
│   └── grafik_5_shap_onem.png
│
├── requirements.txt
└── README.md
```

---

## Gelecek Çalışmalar

- **SMOTE entegrasyonu** — `imbalanced-learn` ile azınlık sınıfı dengelenmesi
- **Uzun eğitim** — Epoch sayısının 50-100 aralığına çıkarılması
- **Hibrit mimari** — XGBoost + BiLSTM birleşik model yapısı
- **LLM entegrasyonu** — FinGPT / BloombergGPT ile metin tabanlı risk analizi
- **Gerçek zamanlı veri akışı** — Dinamik pencere yapısına geçiş

---

## Kaynaklar

Bu proje aşağıdaki çalışmalardan ilham almıştır:

- Yang vd. (2025) — *ResE-BiLSTM Framework for Post-Loan Default Detection* · arXiv:2508.00415
- Chen vd. (2025) — *Enterprise Financial Risk Prediction Based on Deep Learning* · Discover AI
- Yang vd. (2023) — *FinGPT: Open-Source Financial Large Language Models* · arXiv:2306.06031
- Chawla vd. (2002) — *SMOTE: Synthetic Minority Over-sampling Technique* · JAIR

---

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.
