import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# SAYFA YAPISI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NexusCredit · AI Scoring",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e17;
    color: #c9d1e0;
}

/* Kenar çubuğu */
section[data-testid="stSidebar"] {
    background: #0f1623;
    border-right: 1px solid #1e2d45;
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

/* Başlık bloğu */
.hero-block {
    background: linear-gradient(135deg, #0d1b2e 0%, #112240 60%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,149,255,0.12) 0%, transparent 70%);
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #e8f0fe;
    letter-spacing: -0.5px;
    margin: 0;
}
.hero-sub {
    font-size: 0.82rem;
    color: #5a7fa8;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* Metric kartları */
.metric-row { display: flex; gap: 12px; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: #111927;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
.metric-label {
    font-size: 0.7rem; color: #4a6685;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem; font-weight: 600; color: #a8c7fa;
}

/* Karar bloğu */
.verdict-approved {
    background: linear-gradient(135deg, #052a1a 0%, #08391f 100%);
    border: 1px solid #0d6b3a;
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
}
.verdict-rejected {
    background: linear-gradient(135deg, #1f0a0a 0%, #2d1010 100%);
    border: 1px solid #6b1a1a;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
}
.verdict-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem; font-weight: 600; margin: 0;
}
.verdict-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem; font-weight: 600;
}

/* Sebep kartları */
.reason-card {
    background: #111927;
    border: 1px solid #1e2d45;
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.6rem;
}
.reason-title { font-weight: 600; color: #fbbf24; font-size: 0.9rem; }
.reason-body  { color: #8aa4c0; font-size: 0.82rem; margin-top: 0.2rem; }

/* Bölüm başlıkları */
.section-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #3d6090;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    border-bottom: 1px solid #1a2a3e;
    padding-bottom: 6px;
    margin: 1.4rem 0 0.8rem;
}

/* Risk göstergesi */
.risk-bar-wrap {
    background: #111927;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.rbar-track {
    height: 10px;
    background: #1a2a3e;
    border-radius: 6px;
    overflow: hidden;
    margin: 0.6rem 0 0.3rem;
}
.rbar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.5s ease;
}

/* Sidebar elemanlar */
.stSlider > div > div > div { color: #a8c7fa !important; }
label { color: #7a9abf !important; font-size: 0.83rem !important; }
.stNumberInput input, .stTextInput input {
    background: #0f1623 !important;
    border: 1px solid #1e3a5f !important;
    color: #c9d1e0 !important;
    border-radius: 6px !important;
}
button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #1a4a8a, #0d6bcd) !important;
    border: none !important;
    color: white !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.4rem !important;
    width: 100% !important;
}
button[kind="primary"]:hover, .stButton > button:hover {
    background: linear-gradient(135deg, #2160aa, #1480e8) !important;
    box-shadow: 0 0 18px rgba(20, 128, 232, 0.4) !important;
}

/* Plotly arka planı */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL YÜKLEME
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    errors = []
    model  = None
    scaler = None
    model_columns = None

    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('bilstm_credit_model.h5')
    except Exception as e:
        errors.append(f"Model yüklenemedi: {e}")

    try:
        scaler = joblib.load('scaler.gz')
    except Exception as e:
        errors.append(f"Scaler yüklenemedi: {e}")

    try:
        model_columns = joblib.load('model_columns.pkl')
    except Exception as e:
        errors.append(f"model_columns yüklenemedi: {e}")

    return model, scaler, model_columns, errors


model, scaler, model_columns, load_errors = load_assets()

# ─────────────────────────────────────────────
# TAHMIN FONKSİYONU
# ─────────────────────────────────────────────
def predict_credit(age, income, loan_amount, int_rate, employment_years, credit_history_years):
    """
    Hem gerçek model varsa onu hem yoksa kural tabanlı skoru döndürür.
    Dönen değerler: probability (float), prediction (0/1), method (str)
    """
    loan_percent_income = loan_amount / max(income, 1)
    monthly_payment     = (loan_amount * (int_rate / 100 / 12)) / (1 - (1 + int_rate / 100 / 12) ** -60)
    dti_monthly         = monthly_payment / max(income / 12, 1)

    # ── Kural tabanlı çarpanlar ──────────────────────────────
    rule_risk = 0.0
    if loan_percent_income > 0.5:    rule_risk += 0.30
    if loan_percent_income > 1.0:    rule_risk += 0.25
    if loan_percent_income > 2.0:    rule_risk += 0.20
    if income < 15_000:              rule_risk += 0.25
    if income < 5_000:               rule_risk += 0.30
    if int_rate > 18:                rule_risk += 0.15
    if age < 22:                     rule_risk += 0.10
    if employment_years < 1:         rule_risk += 0.15
    if credit_history_years < 2:     rule_risk += 0.10
    if dti_monthly > 0.43:           rule_risk += 0.20
    rule_risk = min(rule_risk, 1.0)

    # ── Model varsa kullan ──────────────────────────────────
    if model is not None and scaler is not None:
        try:
            user_raw = {
                'person_age': age,
                'person_income': income,
                'loan_amnt': loan_amount,
                'loan_int_rate': int_rate,
                'loan_percent_income': loan_percent_income,
                'person_emp_length': employment_years,
                'cb_person_cred_hist_length': credit_history_years,
            }
            # Scaler sütunlarına hizala
            if hasattr(scaler, 'feature_names_in_'):
                expected_cols = list(scaler.feature_names_in_)
            elif model_columns is not None:
                expected_cols = model_columns
            else:
                expected_cols = list(user_raw.keys())

            temp_df = pd.DataFrame(0.0, index=[0], columns=expected_cols)
            for col, val in user_raw.items():
                if col in temp_df.columns:
                    temp_df[col] = val

            scaled_full  = scaler.transform(temp_df)
            # Hedef sütun varsa çıkar
            n_features = scaled_full.shape[1]
            if n_features > len(user_raw):
                scaled_input = scaled_full[:, :-1]
            else:
                scaled_input = scaled_full

            seq_input   = np.repeat(scaled_input[np.newaxis, :, :], 12, axis=1)
            model_prob  = float(model.predict(seq_input, verbose=0)[0][0])

            
            final_prob = 0.55 * model_prob + 0.45 * rule_risk
            threshold  = 0.35  # Daha katı eşik
            prediction = 1 if final_prob > threshold else 0
            return final_prob, prediction, "BiLSTM + Kural Katmanı", loan_percent_income, dti_monthly, monthly_payment

        except Exception as e:
            st.warning(f"Model tahmini başarısız, kural katmanına düşüldü: {e}")

    # ── Sadece kural tabanlı ────────────────────────────────
    threshold  = 0.35
    prediction = 1 if rule_risk > threshold else 0
    return rule_risk, prediction, "Kural Tabanlı Analiz (Model Yok)", loan_percent_income, dti_monthly, monthly_payment


# ─────────────────────────────────────────────
# SIDEBAR — GİRİŞLER
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⬡ NexusCredit")
    st.markdown("<p style='font-size:0.72rem;color:#3d6090;letter-spacing:2px;margin-top:-8px;'>BAŞVURU PANELİ</p>", unsafe_allow_html=True)
    st.divider()

    age               = st.slider("Yaş", 18, 80, 30)
    income            = st.number_input("Yıllık Gelir ($)", min_value=0, value=55_000, step=1000)
    loan_amount       = st.number_input("Kredi Tutarı ($)", min_value=0, value=12_000, step=500)
    int_rate          = st.slider("Faiz Oranı (%)", 3.0, 30.0, 11.5, step=0.5)
    employment_years  = st.slider("Çalışma Süresi (yıl)", 0, 40, 5)
    credit_hist_years = st.slider("Kredi Geçmişi (yıl)", 0, 30, 4)

    st.divider()
    run_btn = st.button("▶  ANALİZİ ÇALIŞTIR", type="primary")

# ─────────────────────────────────────────────
# ANA PANEL
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
  <p class="hero-sub">⬡ NexusCredit Intelligence Platform</p>
  <h1 class="hero-title">Akıllı Kredi Skorlama Sistemi</h1>
  <p style="margin-top:0.6rem;color:#4a6a8a;font-size:0.85rem;">
    BiLSTM Derin Öğrenme · Kural Katmanı · Dinamik Eşik · Açıklanabilir Karar
  </p>
</div>
""", unsafe_allow_html=True)


if load_errors:
    for err in load_errors:
        st.warning(f"⚠ {err}", icon=None)

# ─────── Analiz çalıştırılmadıysa bekleme ekranı ───────
if not run_btn:
    col1, col2, col3 = st.columns(3)
    tiles = [
        ("⬡", "BiLSTM Mimarisi", "Zaman serisi tabanlı derin öğrenme modeli"),
        ("◈", "Kural Katmanı",   "DTI, gelir-borç oranı, faiz filtresi"),
        ("◎", "Açıklanabilir AI","Gerekçeli red/onay kararları"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], tiles):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="min-height:100px;">
                <div style="font-size:1.6rem;margin-bottom:6px;">{icon}</div>
                <div style="font-weight:600;color:#a8c7fa;font-size:0.9rem;">{title}</div>
                <div style="font-size:0.78rem;color:#4a6685;margin-top:4px;">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# ANALİZ BLOĞU
# ─────────────────────────────────────────────
with st.spinner("Model analiz ediyor…"):
    prob, pred, method, lpi, dti_monthly, monthly_pmt = predict_credit(
        age, income, loan_amount, int_rate, employment_years, credit_hist_years
    )

risk_pct   = prob * 100
score_600  = int(850 - prob * 450)   # Simüle kredi skoru (300-850)
dti_pct    = dti_monthly * 100

# ── Metrik Kartları ─────────────────────────────
st.markdown('<p class="section-head">ÖZET METRİKLER</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("Risk Skoru",         f"%{risk_pct:.1f}",                ""),
    ("Simüle Kredi Skoru", f"{score_600}",                    ""),
    ("Borç/Gelir (DTI)",   f"%{lpi*100:.1f}",                 ""),
    ("Aylık Taksit",       f"${monthly_pmt:,.0f}",            ""),
]
for col, (lbl, val, _) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value">{val}</div>
        </div>""", unsafe_allow_html=True)

# ── Karar Bloğu ─────────────────────────────────
st.markdown('<p class="section-head">KARAR</p>', unsafe_allow_html=True)

if pred == 0:
    st.markdown(f"""
    <div class="verdict-approved">
        <p class="verdict-title" style="color:#22c55e;">✓ KREDİ ONAYLANDI</p>
        <p class="verdict-score" style="color:#22c55e;">{risk_pct:.1f}<span style="font-size:1rem;color:#16a34a;">% Risk</span></p>
        <p style="color:#4ade80;font-size:0.82rem;margin-top:4px;">Yöntem: {method}</p>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="verdict-rejected">
        <p class="verdict-title" style="color:#ef4444;">✗ KREDİ REDDEDİLDİ</p>
        <p class="verdict-score" style="color:#ef4444;">{risk_pct:.1f}<span style="font-size:1rem;color:#dc2626;">% Risk</span></p>
        <p style="color:#f87171;font-size:0.82rem;margin-top:4px;">Yöntem: {method}</p>
    </div>""", unsafe_allow_html=True)

# ── Risk Gauge ───────────────────────────────────
bar_color = "#22c55e" if prob < 0.35 else ("#f59e0b" if prob < 0.60 else "#ef4444")
st.markdown('<p class="section-head">RİSK GÖSTERGESI</p>', unsafe_allow_html=True)
st.markdown(f"""
<div class="risk-bar-wrap">
    <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#4a6685;">
        <span>Düşük Risk</span><span>Orta</span><span>Yüksek Risk</span>
    </div>
    <div class="rbar-track">
        <div class="rbar-fill" style="width:{risk_pct:.1f}%;background:{bar_color};"></div>
    </div>
    <div style="text-align:right;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:{bar_color};">
        {risk_pct:.1f}%
    </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GÖRSELLEŞTİRME  
# ─────────────────────────────────────────────
try:
    import plotly.graph_objects as go

    st.markdown('<p class="section-head">KARAR ETKİ FAKTÖRLERİ (NORMALIZE)</p>', unsafe_allow_html=True)

    # Normalize faktör değerleri (0-1 arası)
    factors = {
        "Borç/Gelir Oranı":      min(lpi, 1.0),
        "Faiz Yükü":             (int_rate - 3) / 27,
        "Aylık DTI":             min(dti_monthly, 1.0),
        "Gelir Yeterliliği":     1 - min(income / 100_000, 1.0),
        "Yaş Profili":           abs(age - 40) / 40,
        "Çalışma Süresi Riski":  1 - min(employment_years / 20, 1.0),
        "Kredi Geçmişi Riski":   1 - min(credit_hist_years / 15, 1.0),
    }

    labels = list(factors.keys())
    values = list(factors.values())
    colors = ["#ef4444" if v > 0.6 else "#f59e0b" if v > 0.3 else "#22c55e" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v*100:.0f}%" for v in values],
        textposition='outside',
        textfont=dict(family="IBM Plex Mono", size=11, color="#8aa4c0"),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(11,22,38,0.6)',
        font=dict(family="IBM Plex Sans", color="#8aa4c0", size=11),
        xaxis=dict(
            range=[0, 1.15], showgrid=True,
            gridcolor='rgba(30,45,69,0.8)', tickformat='.0%',
            color='#3d6090'
        ),
        yaxis=dict(color='#8aa4c0', autorange='reversed'),
        margin=dict(l=10, r=60, t=10, b=10),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)
except ImportError:
    st.info("Plotly kurulu değil. `pip install plotly` çalıştırın.")


# ─────────────────────────────────────────────
# RED GEREKÇELERİ
# ─────────────────────────────────────────────
if pred == 1:
    st.markdown('<p class="section-head">RED GEREKÇELERİ & AKILLI ANALİZ</p>', unsafe_allow_html=True)

    reasons = []

    if lpi > 2.0:
        reasons.append((
            "🔴 Kritik Borçlanma Oranı",
            f"Talep edilen kredi tutarı (${loan_amount:,}), yıllık gelirin "
            f"**%{lpi*100:.0f}'i**. Kabul edilebilir üst limit %80'dir."
        ))
    elif lpi > 0.5:
        reasons.append((
            "🟠 Yüksek Borç/Gelir Oranı",
            f"Kredi/gelir oranı %{lpi*100:.1f} ile risk eşiğini (%50) aşıyor."
        ))

    if income < 15_000:
        reasons.append((
            "🔴 Yetersiz Gelir",
            f"Beyan edilen yıllık gelir (${income:,}), bu kredi tutarı için "
            f"asgari gelir eşiğinin altında."
        ))

    if dti_monthly > 0.43:
        reasons.append((
            "🟠 Yüksek Aylık Yük (DTI)",
            f"Aylık taksit (${monthly_pmt:,.0f}), aylık gelirinizin "
            f"**%{dti_pct:.1f}'ini** oluşturuyor. CFPB standartları %43 üzerini riskli sınıflar."
        ))

    if int_rate > 18:
        reasons.append((
            "🟡 Yüksek Faiz Yükü",
            f"%{int_rate:.1f} faiz oranı, toplam geri ödeme maliyetini önemli ölçüde artırıyor."
        ))

    if age < 22 and employment_years < 2:
        reasons.append((
            "🟡 Yetersiz Çalışma & Kredi Geçmişi",
            f"{age} yaş ve {employment_years} yıl çalışma süresiyle kredi geçmişi profili zayıf."
        ))

    if credit_hist_years < 2:
        reasons.append((
            "🟡 Kısa Kredi Geçmişi",
            f"{credit_hist_years} yıllık kredi geçmişi, risk değerlendirmesi için yetersiz."
        ))

    if not reasons:
        reasons.append((
            "🔵 Model Tabanlı Risk",
            "BiLSTM modeli, girilen finansal profilde istatistiksel risk örüntüsü tespit etti."
        ))

    for title, body in reasons:
        st.markdown(f"""
        <div class="reason-card">
            <div class="reason-title">{title}</div>
            <div class="reason-body">{body}</div>
        </div>""", unsafe_allow_html=True)

    # İyileştirme Önerileri
    st.markdown('<p class="section-head">İYİLEŞTİRME ÖNERİLERİ</p>', unsafe_allow_html=True)
    suggestions = []
    if lpi > 0.5:
        ideal_loan = int(income * 0.4)
        suggestions.append(f"💡 Kredi tutarını **${ideal_loan:,}**'e düşürün (yıllık gelirin %40'ı).")
    if income < 40_000:
        suggestions.append("💡 Ek gelir belgesi veya kefil ile başvuruyu güçlendirin.")
    if int_rate > 15:
        suggestions.append("💡 Daha düşük faiz oranlı ürünleri veya güvenceli kredi seçeneklerini değerlendirin.")
    if credit_hist_years < 3:
        suggestions.append("💡 Kredi kartı düzenli kullanımıyla kredi geçmişini güçlendirin.")

    if suggestions:
        for s in suggestions:
            st.markdown(f"<p style='color:#7aa2c8;font-size:0.86rem;margin:4px 0;'>{s}</p>", unsafe_allow_html=True)

elif pred == 0:
    st.markdown('<p class="section-head">ONAY DETAYLARI</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="reason-card" style="border-left-color:#22c55e;">
        <div class="reason-title" style="color:#22c55e;">✓ Finansal Profil Değerlendirmesi</div>
        <div class="reason-body">
        Borç/gelir oranı (%{lpi*100:.1f}), gelir seviyesi (${income:,}/yıl) ve aylık yük (%{dti_pct:.1f}) 
        kabul kriterleri dahilinde. Tahmin yöntemi: <strong>{method}</strong>.
        </div>
    </div>""", unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #1a2a3e;
            text-align:center;font-size:0.72rem;color:#2a4060;letter-spacing:1px;">
    NEXUSCREDIT AI · BiLSTM CREDIT SCORING SYSTEM · DEMO AMAÇLIDIR
</div>""", unsafe_allow_html=True)