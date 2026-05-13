import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────
RUTA_RESULTADOS = "/Users/elenasanjuanmunoz/Documents/TFG/RESULTADOS"

# ─────────────────────────────────────────────
# 1. CARGAR ARRAYS DEL PASO ANTERIOR
# ─────────────────────────────────────────────
X         = np.load(os.path.join(RUTA_RESULTADOS, "X_imagenes.npy"))
y_signos  = np.load(os.path.join(RUTA_RESULTADOS, "y_signos.npy"))
y_grupo   = np.load(os.path.join(RUTA_RESULTADOS, "y_grupo.npy"))
df        = pd.read_csv(os.path.join(RUTA_RESULTADOS, "dataset_procesado.csv"))

print(f"Imágenes cargadas: {X.shape[0]}")
print(f"Distribución de grupos: RDNP={( y_grupo==0).sum()}  RDP={(y_grupo==1).sum()}")

# ─────────────────────────────────────────────
# 2. SPLIT ESTRATIFICADO 70 / 15 / 15
#    Estratificado = mantiene proporción RDNP/RDP en cada partición
# ─────────────────────────────────────────────

# Primero separamos el 70% de entrenamiento del 30% restante
X_train, X_temp, y_signos_train, y_signos_temp, y_grupo_train, y_grupo_temp = train_test_split(
    X, y_signos, y_grupo,
    test_size=0.30,
    random_state=42,        # fija la aleatoriedad para reproducibilidad
    stratify=y_grupo        # mantiene proporción RDNP/RDP
)

# Luego dividimos ese 30% en validación (15%) y prueba (15%)
X_val, X_test, y_signos_val, y_signos_test, y_grupo_val, y_grupo_test = train_test_split(
    X_temp, y_signos_temp, y_grupo_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_grupo_temp
)

print(f"\nSplit realizado:")
print(f"  Entrenamiento: {len(X_train)} imágenes ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validación:    {len(X_val)} imágenes ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Prueba:        {len(X_test)} imágenes ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nDistribución RDNP/RDP por partición:")
for nombre, y in [("Entrenamiento", y_grupo_train), ("Validación", y_grupo_val), ("Prueba", y_grupo_test)]:
    rdnp = (y == 0).sum()
    rdp  = (y == 1).sum()
    print(f"  {nombre}: RDNP={rdnp} ({rdnp/len(y)*100:.1f}%)  RDP={rdp} ({rdp/len(y)*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. DATA AUGMENTATION
#    Solo se aplica al conjunto de entrenamiento
#    Genera versiones modificadas de las imágenes para aumentar la variedad
# ─────────────────────────────────────────────
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(
    rotation_range=15,          # rotaciones aleatorias de hasta 15 grados
    width_shift_range=0.05,     # desplazamiento horizontal leve
    height_shift_range=0.05,    # desplazamiento vertical leve
    zoom_range=0.1,             # zoom aleatorio leve
    horizontal_flip=True,       # volteo horizontal
    vertical_flip=False,        # no volteo vertical (no tiene sentido en retina)
    fill_mode='nearest'         # rellena píxeles nuevos con el vecino más cercano
)

# Sin augmentation para validación y prueba — se usan tal cual
datagen_val  = ImageDataGenerator()
datagen_test = ImageDataGenerator()

print("\nData augmentation configurado para entrenamiento.")
print("Transformaciones: rotación ±15°, desplazamiento 5%, zoom 10%, flip horizontal")

# ─────────────────────────────────────────────
# 4. CREAR GENERADORES
#    En vez de cargar todo en memoria, genera lotes (batches) de imágenes
# ─────────────────────────────────────────────
BATCH_SIZE = 16  # número de imágenes por lote

# Para la tarea de clasificación binaria (RDNP vs RDP)
gen_train_grupo = datagen_train.flow(X_train, y_grupo_train, batch_size=BATCH_SIZE, seed=42)
gen_val_grupo   = datagen_val.flow(X_val,   y_grupo_val,   batch_size=BATCH_SIZE, shuffle=False)
gen_test_grupo  = datagen_test.flow(X_test,  y_grupo_test,  batch_size=BATCH_SIZE, shuffle=False)

# Para la tarea multietiqueta (detección de signos)
gen_train_signos = datagen_train.flow(X_train, y_signos_train, batch_size=BATCH_SIZE, seed=42)
gen_val_signos   = datagen_val.flow(X_val,   y_signos_val,   batch_size=BATCH_SIZE, shuffle=False)
gen_test_signos  = datagen_test.flow(X_test,  y_signos_test,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\nGeneradores creados con batch_size={BATCH_SIZE}")

# ─────────────────────────────────────────────
# 5. VISUALIZAR EFECTO DEL AUGMENTATION
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

imagen_ejemplo = X_train[0:1]  # tomamos la primera imagen del entrenamiento
axes[0].imshow(imagen_ejemplo[0])
axes[0].set_title("Original", fontsize=10)
axes[0].axis("off")

datagen_ejemplo = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

for idx, batch in enumerate(datagen_ejemplo.flow(imagen_ejemplo, batch_size=1)):
    axes[idx + 1].imshow(batch[0])
    axes[idx + 1].set_title(f"Aumentada {idx+1}", fontsize=10)
    axes[idx + 1].axis("off")
    if idx == 3:
        break

plt.suptitle("Efecto del data augmentation sobre una imagen", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_RESULTADOS, "muestra_augmentation.png"), dpi=150)
plt.show()
print("\nFigura guardada en RESULTADOS/muestra_augmentation.png")

# ─────────────────────────────────────────────
# 6. GUARDAR SPLITS PARA EL SIGUIENTE PASO
# ─────────────────────────────────────────────
np.save(os.path.join(RUTA_RESULTADOS, "X_train.npy"),        X_train)
np.save(os.path.join(RUTA_RESULTADOS, "X_val.npy"),          X_val)
np.save(os.path.join(RUTA_RESULTADOS, "X_test.npy"),         X_test)
np.save(os.path.join(RUTA_RESULTADOS, "y_signos_train.npy"), y_signos_train)
np.save(os.path.join(RUTA_RESULTADOS, "y_signos_val.npy"),   y_signos_val)
np.save(os.path.join(RUTA_RESULTADOS, "y_signos_test.npy"),  y_signos_test)
np.save(os.path.join(RUTA_RESULTADOS, "y_grupo_train.npy"),  y_grupo_train)
np.save(os.path.join(RUTA_RESULTADOS, "y_grupo_val.npy"),    y_grupo_val)
np.save(os.path.join(RUTA_RESULTADOS, "y_grupo_test.npy"),   y_grupo_test)

print("\nArrays guardados en TFG/RESULTADOS/")
print("Split completado.")
