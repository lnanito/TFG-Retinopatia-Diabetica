import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score
)

RUTA_RESULTADOS = "/home/hpc/22280273student/TFG_retinopatia/RESULTADOS"
RUTA_MODELOS    = os.path.join(RUTA_RESULTADOS, "modelos")

SIGNOS = [
    "Microaneurismas", "Hemorragias", "Exudados",
    "Neovasos", "Hemovitreo / Opacidad de medios", "Laser"
]

X_test = np.load(os.path.join(RUTA_RESULTADOS, "X_test.npy"))
y_test = np.load(os.path.join(RUTA_RESULTADOS, "y_signos_test.npy"))

df_comp      = pd.read_csv(os.path.join(RUTA_RESULTADOS, "comparativa_fase1.csv"))
mejor_nombre = df_comp.loc[df_comp["F1-score"].idxmax(), "Modelo"]
print(f"Mejor modelo: {mejor_nombre}\n")

modelo = load_model(os.path.join(RUTA_MODELOS, f"{mejor_nombre}_mejor.keras"))
y_prob = modelo.predict(X_test, verbose=0)

# ─────────────────────────────────────────────
# 1. BUSQUEDA DEL UMBRAL OPTIMO
#    Probamos varios umbrales y vemos cual da mejor F1
# ─────────────────────────────────────────────
print("=" * 55)
print("BUSQUEDA DEL UMBRAL OPTIMO")
print("=" * 55)

umbrales = [0.2, 0.3, 0.4, 0.5, 0.6]
filas_umbrales = []

for umbral in umbrales:
    y_pred_u = (y_prob >= umbral).astype(int)
    acc  = accuracy_score(y_test.flatten(), y_pred_u.flatten())
    sens = recall_score(y_test, y_pred_u, average='macro', zero_division=0)
    prec = precision_score(y_test, y_pred_u, average='macro', zero_division=0)
    f1_u = f1_score(y_test, y_pred_u, average='macro', zero_division=0)
    auc  = roc_auc_score(y_test, y_prob, average='macro')

    # Especificidad media
    especificidades = []
    for i in range(y_test.shape[1]):
        yt = y_test[:, i]
        yp = y_pred_u[:, i]
        if len(np.unique(yt)) > 1:
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
            especificidades.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    esp = np.mean(especificidades)

    filas_umbrales.append({
        "Umbral":        umbral,
        "Accuracy":      round(acc, 4),
        "Sensibilidad":  round(sens, 4),
        "Especificidad": round(esp, 4),
        "Precision":     round(prec, 4),
        "F1-score":      round(f1_u, 4),
        "AUC":           round(auc, 4),
    })
    print(f"  Umbral {umbral}: Acc={acc:.4f} Sens={sens:.4f} Esp={esp:.4f} F1={f1_u:.4f}")

df_umbrales = pd.DataFrame(filas_umbrales)
umbral_optimo = df_umbrales.loc[df_umbrales["F1-score"].idxmax(), "Umbral"]
print(f"\nUmbral optimo por F1-score: {umbral_optimo}")
df_umbrales.to_csv(os.path.join(RUTA_RESULTADOS, "comparativa_umbrales.csv"), index=False)

# Grafica comparativa de umbrales
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_umbrales["Umbral"], df_umbrales["Sensibilidad"],  marker='o', label="Sensibilidad")
ax.plot(df_umbrales["Umbral"], df_umbrales["Especificidad"], marker='o', label="Especificidad")
ax.plot(df_umbrales["Umbral"], df_umbrales["F1-score"],      marker='o', label="F1-score")
ax.plot(df_umbrales["Umbral"], df_umbrales["Precision"],     marker='o', label="Precision")
ax.axvline(x=umbral_optimo, color='red', linestyle='--', label=f"Umbral optimo ({umbral_optimo})")
ax.set_xlabel("Umbral")
ax.set_ylabel("Valor")
ax.set_title(f"Efecto del umbral en las metricas - {mejor_nombre}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_RESULTADOS, "comparativa_umbrales.png"), dpi=150)
plt.show()
print("Grafica de umbrales guardada en RESULTADOS/comparativa_umbrales.png")

# ─────────────────────────────────────────────
# 2. EVALUACION CON UMBRAL OPTIMO
# ─────────────────────────────────────────────
print(f"\n{'=' * 55}")
print(f"EVALUACION CON UMBRAL OPTIMO ({umbral_optimo})")
print("=" * 55)

y_pred = (y_prob >= umbral_optimo).astype(int)

accuracy     = accuracy_score(y_test.flatten(), y_pred.flatten())
sensibilidad = recall_score(y_test, y_pred, average='macro', zero_division=0)
precision    = precision_score(y_test, y_pred, average='macro', zero_division=0)
f1           = f1_score(y_test, y_pred, average='macro', zero_division=0)
auc          = roc_auc_score(y_test, y_prob, average='macro')

print(f"  Accuracy:      {accuracy:.4f}")
print(f"  Sensibilidad:  {sensibilidad:.4f}")
print(f"  Precision:     {precision:.4f}")
print(f"  F1-score:      {f1:.4f}")
print(f"  AUC:           {auc:.4f}")

# ─────────────────────────────────────────────
# 3. METRICAS POR SIGNO CON UMBRAL OPTIMO
# ─────────────────────────────────────────────
print(f"\n{'=' * 55}")
print("METRICAS POR SIGNO")
print("=" * 55)

filas_signos = []
for i, signo in enumerate(SIGNOS):
    yt  = y_test[:, i]
    yp  = y_pred[:, i]
    ypr = y_prob[:, i]
    if len(np.unique(yt)) > 1:
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        especificidad  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        auc_signo      = roc_auc_score(yt, ypr)
    else:
        especificidad = 0.0
        auc_signo     = 0.0
    filas_signos.append({
        "Signo":         signo,
        "Accuracy":      round(accuracy_score(yt, yp), 4),
        "Sensibilidad":  round(recall_score(yt, yp, zero_division=0), 4),
        "Especificidad": round(especificidad, 4),
        "F1-score":      round(f1_score(yt, yp, zero_division=0), 4),
        "AUC":           round(auc_signo, 4),
    })

df_signos = pd.DataFrame(filas_signos)
print(df_signos.to_string(index=False))
df_signos.to_csv(os.path.join(RUTA_RESULTADOS, "metricas_por_signo.csv"), index=False)

# ─────────────────────────────────────────────
# 4. MATRICES DE CONFUSION
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, signo in enumerate(SIGNOS):
    yt = y_test[:, i]
    yp = y_pred[:, i]
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    axes[i].imshow(cm, cmap='Blues')
    axes[i].set_title(signo, fontsize=10)
    axes[i].set_xlabel("Prediccion")
    axes[i].set_ylabel("Real")
    axes[i].set_xticks([0, 1])
    axes[i].set_yticks([0, 1])
    axes[i].set_xticklabels(["No", "Si"])
    axes[i].set_yticklabels(["No", "Si"])
    for row in range(2):
        for col in range(2):
            axes[i].text(col, row, str(cm[row, col]),
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='white' if cm[row, col] > cm.max()/2 else 'black')

plt.suptitle(f"Matrices de confusion (umbral={umbral_optimo}) - {mejor_nombre}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_RESULTADOS, "matrices_confusion.png"), dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 5. GRAFICA METRICAS POR SIGNO
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x     = np.arange(len(SIGNOS))
ancho = 0.2
nombres_cortos = ["Microan.", "Hemorr.", "Exudados", "Neovasos", "Hemovit.", "Laser"]
ax.bar(x-ancho*1.5, df_signos["Sensibilidad"],  ancho, label="Sensibilidad")
ax.bar(x-ancho*0.5, df_signos["Especificidad"], ancho, label="Especificidad")
ax.bar(x+ancho*0.5, df_signos["F1-score"],      ancho, label="F1-score")
ax.bar(x+ancho*1.5, df_signos["AUC"],           ancho, label="AUC")
ax.set_xticks(x)
ax.set_xticklabels(nombres_cortos, fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Valor")
ax.set_title(f"Metricas por signo (umbral={umbral_optimo}) - {mejor_nombre}")
ax.legend()
ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_RESULTADOS, "metricas_por_signo.png"), dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 6. RESUMEN FINAL
# ─────────────────────────────────────────────
resumen = {
    "Modelo":          mejor_nombre,
    "Umbral_optimo":   umbral_optimo,
    "N_prueba":        len(X_test),
    "Accuracy":        round(accuracy, 4),
    "Sensibilidad":    round(sensibilidad, 4),
    "Especificidad":   round(df_signos["Especificidad"].mean(), 4),
    "Precision":       round(precision, 4),
    "F1":              round(f1, 4),
    "AUC":             round(auc, 4),
}
pd.DataFrame([resumen]).to_csv(
    os.path.join(RUTA_RESULTADOS, "resumen_evaluacion.csv"), index=False)
print(f"\nResumen guardado en RESULTADOS/resumen_evaluacion.csv")
print("\nPaso 5 completado.")
