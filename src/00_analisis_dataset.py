import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────
RUTA_CSV        = "/Users/elenasanjuanmunoz/Documents/TFG/DATOS/dataset_retinopatia_signos.csv"
RUTA_IMGS       = "/Users/elenasanjuanmunoz/Documents/TFG/IMAGENES"
RUTA_RESULTADOS = "/Users/elenasanjuanmunoz/Documents/TFG/RESULTADOS/analisis_dataset"
CARPETAS        = {"RDNP": "RDNP", "RDP": "RDP", "Sano": "SANO"}

SIGNOS = [
    "Microaneurismas",
    "Hemorragias",
    "Exudados",
    "Neovasos",
    "Hemovitreo / Opacidad de medios",
    "Laser"
]

os.makedirs(RUTA_RESULTADOS, exist_ok=True)

# ─────────────────────────────────────────────
# CARGAR CSV
# ─────────────────────────────────────────────
df = pd.read_csv(RUTA_CSV, sep=';')
print(f"Total imágenes en el dataset: {len(df)}")
print(f"Columnas: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# 1. DISTRIBUCIÓN DE CLASES (Sano / RDNP / RDP)
# ─────────────────────────────────────────────
print("=" * 55)
print("1. DISTRIBUCIÓN DE CLASES")
print("=" * 55)

conteo_grupos = df["Grupo"].value_counts()
print(conteo_grupos.to_string())
print(f"\nTotal: {conteo_grupos.sum()} imágenes")

colores = ["#4CAF50", "#2196F3", "#F44336"]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(conteo_grupos.index, conteo_grupos.values, color=colores, edgecolor='white', linewidth=1.5)
ax.set_title("Distribución de clases en el dataset", fontsize=14, fontweight='bold')
ax.set_xlabel("Grupo")
ax.set_ylabel("Número de imágenes")
for bar, valor in zip(bars, conteo_grupos.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(valor), ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(conteo_grupos.values) * 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_RESULTADOS, "distribucion_clases.png"), dpi=150)
plt.show()
print("Gráfica guardada: distribucion_clases.png\n")

# ─────────────────────────────────────────────
# 2. DISTRIBUCIÓN DE SIGNOS (total y por grupo)
# ─────────────────────────────────────────────
print("=" * 55)
print("2. DISTRIBUCIÓN DE SIGNOS")
print("=" * 55)

# Solo imágenes con retinopatía (RDNP + RDP)
df_rd = df[df["Grupo"].isin(["RDNP", "RDP"])].copy()

print("\nPresencia de cada signo en el dataset completo con retinopatía:")
for signo in SIGNOS:
    if signo in df_rd.columns:
        n = int(df_rd[signo].sum())
        pct = n / len(df_rd) * 100
        print(f"  {signo:40s}: {n:4d} imágenes ({pct:.1f}%)")

# Gráfica total
conteo_signos = {s: int(df_rd[s].sum()) for s in SIGNOS if s in df_rd.columns}
fig, ax = plt.subplots(figsize=(10, 5))
nombres_cortos = ["Microan.", "Hemorr.", "Exudados", "Neovasos", "Hemovit.", "Laser"]
bars = ax.bar(nombres_cortos, conteo_signos.values(), color="#2196F3", edgecolor='white', linewidth=1.5)
ax.set_title("Número de imágenes con cada signo clínico", fontsize=14, fontweight='bold')
ax.set_ylabel("Número de imágenes")
for bar, valor in zip(bars, conteo_signos.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(valor), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(conteo_signos.values()) * 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_RESULTADOS, "distribucion_signos_total.png"), dpi=150)
plt.show()
print("Gráfica guardada: distribucion_signos_total.png")

# Gráfica por grupo (RDNP vs RDP)
print("\nPresencia de signos por grupo:")
conteo_por_grupo = {}
for grupo in ["RDNP", "RDP"]:
    df_g = df[df["Grupo"] == grupo]
    conteo_por_grupo[grupo] = {s: int(df_g[s].sum()) for s in SIGNOS if s in df_g.columns}
    print(f"\n  {grupo}:")
    for signo, n in conteo_por_grupo[grupo].items():
        pct = n / len(df_g) * 100
        print(f"    {signo:40s}: {n:4d} ({pct:.1f}%)")

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(nombres_cortos))
ancho = 0.35
bars1 = ax.bar(x - ancho/2, conteo_por_grupo["RDNP"].values(), ancho, label="RDNP", color="#2196F3", edgecolor='white')
bars2 = ax.bar(x + ancho/2, conteo_por_grupo["RDP"].values(),  ancho, label="RDP",  color="#F44336", edgecolor='white')
ax.set_title("Distribución de signos por grupo (RDNP vs RDP)", fontsize=14, fontweight='bold')
ax.set_ylabel("Número de imágenes")
ax.set_xticks(x)
ax.set_xticklabels(nombres_cortos)
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                str(int(h)), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_RESULTADOS, "distribucion_signos_por_grupo.png"), dpi=150)
plt.show()
print("\nGráfica guardada: distribucion_signos_por_grupo.png\n")

# ─────────────────────────────────────────────
# 3. CO-OCURRENCIA DE SIGNOS
# ─────────────────────────────────────────────
print("=" * 55)
print("3. CO-OCURRENCIA DE SIGNOS")
print("=" * 55)

signos_presentes = [s for s in SIGNOS if s in df_rd.columns]
matriz_cooc = pd.DataFrame(index=signos_presentes, columns=signos_presentes, dtype=float)

for s1 in signos_presentes:
    for s2 in signos_presentes:
        cooc = int(((df_rd[s1] == 1) & (df_rd[s2] == 1)).sum())
        matriz_cooc.loc[s1, s2] = cooc

print("\nMatriz de co-ocurrencia (número de imágenes con ambos signos):")
print(matriz_cooc.to_string())

nombres_cortos_full = ["Microan.", "Hemorr.", "Exudados", "Neovasos", "Hemovit.", "Laser"]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    matriz_cooc.astype(float),
    annot=True, fmt='.0f',
    cmap='Blues',
    xticklabels=nombres_cortos_full,
    yticklabels=nombres_cortos_full,
    linewidths=0.5,
    ax=ax
)
ax.set_title("Co-ocurrencia de signos clínicos", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RUTA_RESULTADOS, "coocurrencia_signos.png"), dpi=150)
plt.show()
print("Gráfica guardada: coocurrencia_signos.png\n")

# ─────────────────────────────────────────────
# 4. NÚMERO DE SIGNOS POR IMAGEN
# ─────────────────────────────────────────────
print("=" * 55)
print("4. NÚMERO DE SIGNOS POR IMAGEN")
print("=" * 55)

signos_cols = [s for s in SIGNOS if s in df_rd.columns]
df_rd = df_rd.copy()
df_rd["n_signos"] = df_rd[signos_cols].sum(axis=1)

conteo_n_signos = df_rd["n_signos"].value_counts().sort_index()
print("\nDistribución del número de signos por imagen:")
for n, count in conteo_n_signos.items():
    pct = count / len(df_rd) * 100
    print(f"  {int(n)} signo(s): {count} imágenes ({pct:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(conteo_n_signos.index.astype(int), conteo_n_signos.values,
              color="#9C27B0", edgecolor='white', linewidth=1.5)
ax.set_title("Número de signos clínicos por imagen", fontsize=14, fontweight='bold')
ax.set_xlabel("Número de signos")
ax.set_ylabel("Número de imágenes")
ax.set_xticks(conteo_n_signos.index.astype(int))
for bar, valor in zip(bars, conteo_n_signos.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(valor), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_RESULTADOS, "n_signos_por_imagen.png"), dpi=150)
plt.show()
print("Gráfica guardada: n_signos_por_imagen.png\n")

# ─────────────────────────────────────────────
# 5. CALIDAD Y TAMAÑO ORIGINAL DE LAS IMÁGENES
# ─────────────────────────────────────────────
print("=" * 55)
print("5. TAMAÑO ORIGINAL DE LAS IMÁGENES")
print("=" * 55)

anchos, altos = [], []
errores = 0
muestra = df.sample(min(200, len(df)), random_state=42)

for _, row in muestra.iterrows():
    carpeta = CARPETAS.get(row["Grupo"], "")
    ruta = os.path.join(RUTA_IMGS, carpeta, row["Imagen"])
    try:
        img = Image.open(ruta)
        w, h = img.size
        anchos.append(w)
        altos.append(h)
    except:
        errores += 1

print(f"\nImágenes analizadas: {len(anchos)} (errores: {errores})")
if anchos:
    print(f"Ancho  — mín: {min(anchos)}px  máx: {max(anchos)}px  media: {int(np.mean(anchos))}px")
    print(f"Alto   — mín: {min(altos)}px  máx: {max(altos)}px  media: {int(np.mean(altos))}px")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(anchos, bins=20, color="#FF9800", edgecolor='white')
    axes[0].set_title("Distribución de anchuras originales", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Píxeles")
    axes[0].set_ylabel("Número de imágenes")
    axes[0].axvline(x=224, color='red', linestyle='--', label="Tamaño objetivo (224px)")
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].hist(altos, bins=20, color="#FF9800", edgecolor='white')
    axes[1].set_title("Distribución de alturas originales", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Píxeles")
    axes[1].set_ylabel("Número de imágenes")
    axes[1].axvline(x=224, color='red', linestyle='--', label="Tamaño objetivo (224px)")
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_RESULTADOS, "tamanos_originales.png"), dpi=150)
    plt.show()
    print("Gráfica guardada: tamanos_originales.png\n")

# ─────────────────────────────────────────────
# 6. RESUMEN GENERAL DEL DATASET
# ─────────────────────────────────────────────
print("=" * 55)
print("6. RESUMEN GENERAL")
print("=" * 55)

resumen = {
    "Total imágenes": len(df),
    "Imágenes sanas": int((df["Grupo"] == "Sano").sum()),
    "Imágenes RDNP": int((df["Grupo"] == "RDNP").sum()),
    "Imágenes RDP": int((df["Grupo"] == "RDP").sum()),
}
for signo in signos_cols:
    resumen[f"Con {signo}"] = int(df_rd[signo].sum())

df_resumen = pd.DataFrame([resumen]).T
df_resumen.columns = ["Valor"]
print(df_resumen.to_string())
df_resumen.to_csv(os.path.join(RUTA_RESULTADOS, "resumen_dataset.csv"))
print("\nResumen guardado: resumen_dataset.csv")
print("\nAnálisis completado. Todas las gráficas en:", RUTA_RESULTADOS)
