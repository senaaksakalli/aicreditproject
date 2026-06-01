import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import tensorflow as tf
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
import xgboost as xgb
import lightgbm as lgb

PALETTE = {
    "navy" : "#1B3A6B", "teal"  : "#0D7377",
    "green": "#1A7A4A", "red"   : "#C0392B",
    "orange":"#E67E22", "gray"  : "#7F8C8D", "lgray": "#ECF0F1",
}

MODEL_COLORS = {
    "Lojistik Reg.": "#3498DB", "Random Forest": "#2ECC71",
    "XGBoost"      : "#E74C3C", "LightGBM"     : "#9B59B6",
    "MLP"          : "#F39C12", "BiLSTM"       : "#1B3A6B",
}

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("  FİNANSAL RİSK YÖNETİMİ — ANALİZ BAŞLIYOR")
print("=" * 60)

# ── 1. VERİ ─────────────────────────────────────────────────
print("\n[1/8] Veri hazırlanıyor...")
df     = pd.read_csv("cleaned_credit_data.csv")
df_enc = pd.get_dummies(df)
target = "loan_status"

X_raw  = df_enc.drop(target, axis=1).values
y_raw  = df_enc[target].values
feat_cols = df_enc.drop(target, axis=1).columns.tolist()

X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

sc_flat = MinMaxScaler()
X_tr_sc = sc_flat.fit_transform(X_tr)
X_te_sc = sc_flat.transform(X_te)

n_pos = int(y_tr.sum())
n_neg = len(y_tr) - n_pos
cw    = n_neg / n_pos

df_seq = pd.get_dummies(pd.read_csv("cleaned_credit_data.csv"))
sc_seq = MinMaxScaler()
scaled_seq = sc_seq.fit_transform(df_seq)

def make_sequences(data, n=12):
    Xs, ys = [], []
    for i in range(len(data) - n):
        Xs.append(data[i:i+n, :-1])
        ys.append(data[i+n, -1])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_sequences(scaled_seq, 12)
Xtr_seq, Xte_seq, ytr_seq, yte_seq = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)

print(f"   Toplam kayıt   : {len(df):,}")
print(f"   Eğitim (flat)  : {len(X_tr):,}  |  Test: {len(X_te):,}")
print(f"   Temerrüt oranı : %{y_raw.mean()*100:.1f}  (class weight ≈ {cw:.2f})")
print(f"   Özellik sayısı : {len(feat_cols)}")

# ── 2. EĞİTİM ───────────────────────────────────────────────
print("\n[2/8] Modeller eğitiliyor...")
models_flat = {
    "Lojistik Reg.": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
    "XGBoost"      : xgb.XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0, scale_pos_weight=cw, n_estimators=200),
    "LightGBM"     : lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight="balanced", n_estimators=200),
    "MLP"          : MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True),
}

for name, m in models_flat.items():
    m.fit(X_tr_sc, y_tr)
    print(f"   ✓  {name} hazır")

bilstm = tf.keras.models.load_model("bilstm_credit_model.h5")
print(f"   ✓  BiLSTM yüklendi ({bilstm.count_params():,} parametre)")

# ── 3. METRİKLER ────────────────────────────────────────────
print("\n[3/8] Metrikler hesaplanıyor...")
results = {}
for name, m in models_flat.items():
    yp  = m.predict(X_te_sc)
    ypr = m.predict_proba(X_te_sc)[:, 1]
    results[name] = dict(
        y_pred=yp, y_prob=ypr, y_test=y_te,
        acc =accuracy_score(y_te, yp),
        prec=precision_score(y_te, yp, zero_division=0),
        rec =recall_score(y_te, yp, zero_division=0),
        f1  =f1_score(y_te, yp, zero_division=0),
        mcc =matthews_corrcoef(y_te, yp),
        auc =roc_auc_score(y_te, ypr),
        pr  =average_precision_score(y_te, ypr),
        cm  =confusion_matrix(y_te, yp),
    )

bpr  = bilstm.predict(Xte_seq, verbose=0).flatten()
bprd = (bpr > 0.5).astype(int)
results["BiLSTM"] = dict(
    y_pred=bprd, y_prob=bpr, y_test=yte_seq,
    acc =accuracy_score(yte_seq, bprd),
    prec=precision_score(yte_seq, bprd, zero_division=0),
    rec =recall_score(yte_seq, bprd, zero_division=0),
    f1  =f1_score(yte_seq, bprd, zero_division=0),
    mcc =matthews_corrcoef(yte_seq, bprd),
    auc =roc_auc_score(yte_seq, bpr),
    pr  =average_precision_score(yte_seq, bpr),
    cm  =confusion_matrix(yte_seq, bprd),
)

rows = [{
    "Model"    : n,
    "Accuracy" : round(r["acc"],  4),
    "Precision": round(r["prec"], 4),
    "Recall"   : round(r["rec"],  4),
    "F1-Score" : round(r["f1"],   4),
    "MCC"      : round(r["mcc"],  4),
    "ROC-AUC"  : round(r["auc"],  4),
    "PR-AUC"   : round(r["pr"],   4),
} for n, r in results.items()]

tablo = pd.DataFrame(rows)
tablo.to_csv(f"{OUT}/tablo_performans.csv", index=False, encoding="utf-8-sig")
print("\n  ┌─ PERFORMANS METRİKLERİ " + "─"*58)
print(tablo.to_string(index=False))
print("  └" + "─"*84)

# ── 4. CONFUSION MATRIX ──────────────────────────────────────
print("\n[4/8] Confusion Matrix grafikleri oluşturuluyor...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Karmaşıklık Matrisi — Tüm Modeller\n(Satır: Gerçek Sınıf  |  Sütun: Tahmin Edilen Sınıf)", fontsize=15, fontweight="bold", color=PALETTE["navy"], y=1.01)
labels = ["Geri Ödendi (0)", "Temerrüt (1)"]

for idx, (name, r) in enumerate(results.items()):
    ax = axes.flatten()[idx]
    cm = r["cm"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot  = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i,j] = f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)"
    tn, fp, fn, tp = cm.ravel()
    cmap = "YlOrRd" if name == "BiLSTM" else "Blues"
    sns.heatmap(cm, annot=annot, fmt="", ax=ax, cmap=cmap, linewidths=0.5, xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size":11, "weight":"bold"})
    clr = PALETTE["red"] if name == "BiLSTM" else PALETTE["navy"]
    ax.set_title(f"{name}\nF1={r['f1']:.3f}  |  MCC={r['mcc']:.3f}", fontsize=12, fontweight="bold", color=clr)
    ax.set_xlabel("Tahmin Edilen", fontsize=10)
    ax.set_ylabel("Gerçek", fontsize=10)
    ax.text(0.5, -0.22, f"TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}", ha="center", transform=ax.transAxes, fontsize=9, color=PALETTE["gray"])

plt.tight_layout(pad=2.5)
plt.savefig(f"{OUT}/grafik_1_confusion_matrix.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()

# ── 5. ROC EĞRİSİ ───────────────────────────────────────────
print("\n[5/8] ROC eğrisi çiziliyor...")
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot([0,1],[0,1],"k--",lw=1.2,alpha=0.6,label="Rastgele Sınıflandırıcı (AUC=0.500)")
for name, r in results.items():
    fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
    ax.plot(fpr, tpr, lw=(3.0 if name == "BiLSTM" else 1.8), alpha=(1.0 if name == "BiLSTM" else 0.85), color=MODEL_COLORS[name], label=f"{name}  (AUC = {r['auc']:.3f})")
ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=13)
ax.set_ylabel("Doğru Pozitif Oranı (TPR)", fontsize=13)
ax.set_title("ROC Eğrisi — Model Karşılaştırması", fontsize=14, fontweight="bold", color=PALETTE["navy"])
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig(f"{OUT}/grafik_2_roc_egrisi.png", dpi=180, bbox_inches="tight")
plt.close()

# ── 6. PR EĞRİSİ ────────────────────────────────────────────
print("\n[6/8] PR eğrisi çiziliyor...")
fig, ax = plt.subplots(figsize=(10, 8))
for name, r in results.items():
    p_arr, r_arr, _ = precision_recall_curve(r["y_test"], r["y_prob"])
    ax.plot(r_arr, p_arr, lw=(3.0 if name == "BiLSTM" else 1.8), color=MODEL_COLORS[name], label=f"{name}  (PR-AUC = {r['pr']:.3f})")
ax.set_xlabel("Recall (Duyarlılık)", fontsize=13)
ax.set_ylabel("Precision (Kesinlik)", fontsize=13)
ax.set_title("Precision-Recall Eğrisi", fontsize=14, fontweight="bold", color=PALETTE["navy"])
ax.legend(loc="upper right", fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig(f"{OUT}/grafik_3_pr_egrisi.png", dpi=180, bbox_inches="tight")
plt.close()

# ── 7. KARŞILAŞTIRMA ÇUBUKLARI ──────────────────────────────
print("\n[7/8] Karşılaştırma grafikleri oluşturuluyor...")
# Tam burada listeleri zorunlu olarak tekrar tanımlıyoruz
metriks = ["Accuracy","Precision","Recall","F1-Score","MCC","ROC-AUC","PR-AUC"]
model_names = list(results.keys())

fig, axes = plt.subplots(3, 3, figsize=(20, 16))
axes = axes.flatten()
for i, metrik in enumerate(metriks):
    ax   = axes[i]
    vals = tablo[metrik].values
    clrs = [MODEL_COLORS[n] for n in model_names]
    bars = ax.barh(model_names, vals, color=clrs, edgecolor="white", linewidth=0.8, height=0.65)
    bi   = model_names.index("BiLSTM")
    bars[bi].set_edgecolor(PALETTE["red"])
    bars[bi].set_linewidth(2.5)
    ax.set_title(metrik, fontsize=12, fontweight="bold", color=PALETTE["navy"])
    ax.set_xlim(0, 1.15); ax.grid(axis="x", alpha=0.3)
for j in range(len(metriks), len(axes)): axes[j].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/grafik_4_model_karsilastirma.png", dpi=180)
plt.close()

# ── 8. SHAP ÖNEMİ ───────────────────────────────────────────
print("\n[8/8] SHAP özellikleri hesaplanıyor...")
try:
    import shap
    explainer = shap.TreeExplainer(models_flat["XGBoost"])
    sv = explainer.shap_values(X_te_sc[:600])
    shap_mean = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({"Özellik": feat_cols, "SHAP": shap_mean}).sort_values("SHAP", ascending=True).tail(15)
    fig, ax_sh = plt.subplots(figsize=(10, 8))
    ax_sh.barh(shap_df["Özellik"], shap_df["SHAP"], color=PALETTE["teal"])
    ax_sh.set_title("SHAP Özellik Önemi Analizi (XGBoost)")
    plt.savefig(f"{OUT}/grafik_5_shap_onem.png", dpi=180, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"   ⚠  SHAP atlandı: {e}")

print("\n" + "="*60)
print("  ANALİZ BAŞARIYLA TAMAMLANDI!")
print("=" * 60)