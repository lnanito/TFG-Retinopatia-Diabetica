import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────
RUTA_CSV       = "/Users/elenasanjuanmunoz/Documents/TFG/DATOS/dataset_retinopatia_signos.csv"
RUTA_IMGS = "/Users/elenasanjuanmunoz/Documents/TFG/IMAGENES"
RUTA_RESULTADOS = "/Users/elenasanjuanmunoz/Documents/TFG/RESULTADOS"
CARPETAS = {"RDNP": "RDNP", "RDP": "RDP", "Sano": "SANO"}

IMG_SIZE       = (224, 224)  # tamaño que esperan ResNet, DenseNet, Inception

# ─────────────────────────────────────────────
# 1. CARGAR EL CSV
# ─────────────────────────────────────────────
df = pd.read_csv(RUTA_CSV, sep=';')
print(f"Total filas en CSV: {len(df)}")
print(f"Columnas: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# 2. VERIFICAR QUE LAS IMÁGENES EXISTEN EN DISCO
# ─────────────────────────────────────────────
def ruta_imagen(row):
    carpeta = CARPETAS.get(row["Grupo"], "")
    return os.path.join(RUTA_IMGS, carpeta, row["Imagen"])

df["ruta"] = df.apply(ruta_imagen, axis=1)
df["existe"] = df["ruta"].apply(os.path.exists)

n_ok      = df["existe"].sum()
n_falta   = (~df["existe"]).sum()
print(f"Imágenes encontradas: {n_ok}")
print(f"Imágenes NO encontradas: {n_falta}")

if n_falta > 0:
    print("\nImágenes que faltan:")
    print(df[~df["existe"]][["Imagen", "Grupo"]].to_string(index=False))

# ─────────────────────────────────────────────
# 3. FUNCIÓN DE CARGA Y PREPROCESAMIENTO
# ─────────────────────────────────────────────
def cargar_imagen(ruta, size=IMG_SIZE):
    """
    Carga una imagen, la redimensiona y normaliza los píxeles a [0, 1].
    Devuelve un array numpy de shape (224, 224, 3).
    """
    img = Image.open(ruta).convert("RGB")   # asegura 3 canales (RGB)
    img = img.resize(size)                  # redimensiona a 224x224
    arr = np.array(img) / 255.0             # normaliza de [0,255] a [0,1]
    return arr

# ─────────────────────────────────────────────
# 4. CARGAR TODAS LAS IMÁGENES QUE EXISTEN
# ─────────────────────────────────────────────
df_ok = df[df["existe"]].reset_index(drop=True)

imagenes = []
errores  = []

for _, row in df_ok.iterrows():
    try:
        arr = cargar_imagen(row["ruta"])
        imagenes.append(arr)
    except Exception as e:
        errores.append((row["Imagen"], str(e)))
        imagenes.append(None)

print(f"\nImágenes cargadas correctamente: {len(imagenes) - len(errores)}")
if errores:
    print(f"Errores al cargar: {len(errores)}")
    for nombre, err in errores:
        print(f"  {nombre}: {err}")

# ─────────────────────────────────────────────
# 5. CONVERTIR A ARRAYS NUMPY
# ─────────────────────────────────────────────
# Filtramos las que fallaron (None)
indices_ok  = [i for i, img in enumerate(imagenes) if img is not None]
X           = np.array([imagenes[i] for i in indices_ok])  # shape: (N, 224, 224, 3)
df_final    = df_ok.iloc[indices_ok].reset_index(drop=True)

print(f"\nShape del array de imágenes: {X.shape}")
print(f"Valor mínimo de píxel: {X.min():.3f}  (debería ser ~0.0)")
print(f"Valor máximo de píxel: {X.max():.3f}  (debería ser ~1.0)")

# ─────────────────────────────────────────────
# 6. EXTRAER ETIQUETAS
# ─────────────────────────────────────────────
SIGNOS = [
    "Microaneurismas",
    "Hemorragias",
    "Exudados",
    "Neovasos",
    "Hemovitreo / Opacidad de medios",
    "Laser"
]

y_signos     = df_final[SIGNOS].values.astype(np.float32)   # shape: (N, 6)  — multietiqueta
y_grupo      = (df_final["Grupo"] == "RDP").astype(int).values  # 1=RDP, 0=RDNP
y_retinopatia = df_final["Retinopatia"].values.astype(np.float32)  # todo 1s por ahora

print(f"\nShape etiquetas signos:      {y_signos.shape}")
print(f"Shape etiqueta grupo:        {y_grupo.shape}")
print(f"Shape etiqueta retinopatia:  {y_retinopatia.shape}")

# ─────────────────────────────────────────────
# 7. VISUALIZACIÓN DE MUESTRA 
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.imshow(X[i])
    grupo  = df_final.loc[i, "Grupo"]
    signos_presentes = [s for s in SIGNOS if df_final.loc[i, s] == 1]
    titulo = f"{grupo}\n{', '.join(signos_presentes) if signos_presentes else 'sin signos'}"
    ax.set_title(titulo, fontsize=7)
    ax.axis("off")

plt.suptitle("Muestra de imágenes preprocesadas (224×224, normalizadas)", fontsize=11)
plt.tight_layout()
os.makedirs(RUTA_RESULTADOS, exist_ok=True)
plt.savefig(os.path.join(RUTA_RESULTADOS, "muestra_preprocesamiento.png"), dpi=150)
plt.show()
print("\nFigura guardada en RESULTADOS/muestra_preprocesamiento.png")

# ─────────────────────────────────────────────
# 8. GUARDAR ARRAYS PARA EL SIGUIENTE PASO
# ─────────────────────────────────────────────
np.save(os.path.join(RUTA_RESULTADOS, "X_imagenes.npy"), X)
np.save(os.path.join(RUTA_RESULTADOS, "y_signos.npy"),   y_signos)
np.save(os.path.join(RUTA_RESULTADOS, "y_grupo.npy"),    y_grupo)
df_final.to_csv(os.path.join(RUTA_RESULTADOS, "dataset_procesado.csv"), index=False)

print("\nArchivos guardados en TFG/RESULTADOS/:")
print("  X_imagenes.npy        — imágenes preprocesadas")
print("  y_signos.npy          — etiquetas multietiqueta (6 signos)")
print("  y_grupo.npy           — etiqueta RDNP/RDP")
print("  dataset_procesado.csv — CSV con rutas y etiquetas limpias")
print("\nPreprocesamiento completado.")
