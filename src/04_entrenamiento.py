import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────
RUTA_RESULTADOS = "/Users/elenasanjuanmunoz/Documents/TFG/RESULTADOS"
RUTA_MODELOS    = os.path.join(RUTA_RESULTADOS, "modelos")

SIGNOS = [
    "Microaneurismas",
    "Hemorragias",
    "Exudados",
    "Neovasos",
    "Hemovitreo / Opacidad de medios",
    "Laser"
]

# ─────────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────────
X_train = np.load(os.path.join(RUTA_RESULTADOS, "X_train.npy"))
X_val   = np.load(os.path.join(RUTA_RESULTADOS, "X_val.npy"))
X_test  = np.load(os.path.join(RUTA_RESULTADOS, "X_test.npy"))
y_train = np.load(os.path.join(RUTA_RESULTADOS, "y_signos_train.npy"))
y_val   = np.load(os.path.join(RUTA_RESULTADOS, "y_signos_val.npy"))
y_test  = np.load(os.path.join(RUTA_RESULTADOS, "y_signos_test.npy"))
X_todos = np.load(os.path.join(RUTA_RESULTADOS, "X_imagenes.npy"))
y_todos = np.load(os.path.join(RUTA_RESULTADOS, "y_signos.npy"))
y_grupo = np.load(os.path.join(RUTA_RESULTADOS, "y_grupo.npy"))

print(f"Datos cargados:")
print(f"  Entrenamiento: {X_train.shape[0]} imagenes")
print(f"  Validacion:    {X_val.shape[0]} imagenes")
print(f"  Prueba:        {X_test.shape[0]} imagenes\n")

# ─────────────────────────────────────────────
# 2. CONFIGURACION
# ─────────────────────────────────────────────
BATCH_SIZE = 16
EPOCHS     = 50

datagen_train = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen_val = ImageDataGenerator()

# ─────────────────────────────────────────────
# 3. CALLBACKS
# ─────────────────────────────────────────────
def crear_callbacks(nombre_modelo):
    ruta_mejor = os.path.join(RUTA_MODELOS, f"{nombre_modelo}_mejor.keras")
    return [
        ModelCheckpoint(
            filepath=ruta_mejor,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

# ─────────────────────────────────────────────
# 4. FUNCION DE METRICAS
#    Calcula todas las metricas descritas en metodologia
# ─────────────────────────────────────────────
def calcular_metricas(y_real, y_pred, y_prob):
    """
    Calcula accuracy, sensibilidad, especificidad, F1 y AUC
    para clasificacion multietiqueta (promedio entre signos)
    """
    accuracy     = accuracy_score(y_real.flatten(), y_pred.flatten())
    sensibilidad = recall_score(y_real, y_pred, average='macro', zero_division=0)
    f1           = f1_score(y_real, y_pred, average='macro', zero_division=0)
    auc          = roc_auc_score(y_real, y_prob, average='macro')

    # Especificidad = promedio de especificidades por signo
    especificidades = []
    for i in range(y_real.shape[1]):
        yt = y_real[:, i]
        yp = y_pred[:, i]
        if len(np.unique(yt)) > 1:
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            especificidades.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    especificidad = np.mean(especificidades) if especificidades else 0.0

    return {
        "Accuracy":      round(accuracy, 4),
        "Sensibilidad":  round(sensibilidad, 4),
        "Especificidad": round(especificidad, 4),
        "F1-score":      round(f1, 4),
        "AUC":           round(auc, 4),
    }

# ─────────────────────────────────────────────
# 5. FASE 1 — COMPARACION INICIAL DE LOS TRES MODELOS
# ─────────────────────────────────────────────
print("=" * 55)
print("FASE 1 — Comparacion inicial de los tres modelos")
print("=" * 55)

modelos_info = [
    ("resnet50_signos.keras",    "ResNet50"),
    ("densenet121_signos.keras", "DenseNet121"),
    ("inceptionv3_signos.keras", "InceptionV3"),
]

historias       = {}
resultados_val  = {}
resultados_test = {}

for archivo, nombre in modelos_info:
    print(f"\nEntrenando {nombre}...")

    modelo = load_model(os.path.join(RUTA_MODELOS, archivo))

    gen_train = datagen_train.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=42)
    gen_val   = datagen_val.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

    historia = modelo.fit(
        gen_train,
        epochs=EPOCHS,
        validation_data=gen_val,
        callbacks=crear_callbacks(nombre),
        verbose=1
    )

    historias[nombre] = historia.history

    # Metricas en validacion
    resultados_val[nombre] = {
        "val_loss":     round(min(historia.history["val_loss"]), 4),
        "val_accuracy": round(max(historia.history["val_accuracy"]), 4),
        "val_auc":      round(max(historia.history["val_auc"]), 4),
        "epocas":       len(historia.history["val_loss"])
    }

    # Metricas en prueba — evaluacion con datos que el modelo no ha visto
    print(f"  Evaluando {nombre} en conjunto de prueba...")
    y_prob = modelo.predict(X_test, verbose=0)
    y_pred = (y_prob >= 0.5).astype(int)
    metricas_test = calcular_metricas(y_test, y_pred, y_prob)
    resultados_test[nombre] = metricas_test

    print(f"  {nombre} — Prueba:")
    print(f"    Accuracy={metricas_test['Accuracy']}  "
          f"Sensibilidad={metricas_test['Sensibilidad']}  "
          f"Especificidad={metricas_test['Especificidad']}  "
          f"F1={metricas_test['F1-score']}  "
          f"AUC={metricas_test['AUC']}")

# ─────────────────────────────────────────────
# 6. TABLA COMPARATIVA DE LOS TRES MODELOS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("COMPARATIVA DE LOS TRES MODELOS — CONJUNTO DE PRUEBA")
print("=" * 55)

filas = []
for nombre, m in resultados_test.items():
    filas.append({"Modelo": nombre, **m})

df_comparativa = pd.DataFrame(filas)
print(df_comparativa.to_string(index=False))

# El mejor modelo es el que tiene mayor F1-score en prueba
mejor_nombre = max(resultados_test, key=lambda x: resultados_test[x]["F1-score"])
print(f"\nMejor modelo: {mejor_nombre} "
      f"(F1={resultados_test[mejor_nombre]['F1-score']})")

df_comparativa.to_csv(
    os.path.join(RUTA_RESULTADOS, "comparativa_fase1.csv"), index=False
)
print("Comparativa guardada en RESULTADOS/comparativa_fase1.csv")

# ─────────────────────────────────────────────
# 7. CURVAS DE ENTRENAMIENTO
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 16))

for idx, (nombre, h) in enumerate(historias.items()):
    epocas = range(1, len(h["loss"]) + 1)

    axes[idx, 0].plot(epocas, h["loss"],     label="Entrenamiento")
    axes[idx, 0].plot(epocas, h["val_loss"], label="Validacion")
    axes[idx, 0].set_title(f"{nombre} — Perdida")
    axes[idx, 0].set_xlabel("Epoca")
    axes[idx, 0].legend()

    axes[idx, 1].plot(epocas, h["accuracy"],     label="Entrenamiento")
    axes[idx, 1].plot(epocas, h["val_accuracy"], label="Validacion")
    axes[idx, 1].set_title(f"{nombre} — Accuracy")
    axes[idx, 1].set_xlabel("Epoca")
    axes[idx, 1].legend()

plt.suptitle("Curvas de entrenamiento — Fase 1", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_RESULTADOS, "curvas_entrenamiento_fase1.png"), dpi=150)
plt.show()
print("Curvas guardadas en RESULTADOS/curvas_entrenamiento_fase1.png")

# ─────────────────────────────────────────────
# 8. FASE 2 — ENTRENAMIENTO ESTABLE DEL MEJOR MODELO
#    Se entrena varias veces con distintas particiones
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"FASE 2 — Entrenamiento estable de {mejor_nombre}")
print("=" * 55)

N_REPETICIONES = 5
semillas       = [42, 7, 13, 21, 99]
metricas_fase2 = []

for i, semilla in enumerate(semillas):
    print(f"\nRepeticion {i+1}/{N_REPETICIONES} (semilla={semilla})...")

    X_tr, X_temp, y_tr, y_temp, yg_tr, yg_temp = train_test_split(
        X_todos, y_todos, y_grupo,
        test_size=0.30,
        random_state=semilla,
        stratify=y_grupo
    )
    X_v, X_te, y_v, y_te = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=semilla
    )

    archivo_mejor = f"{mejor_nombre}_mejor.keras"
    modelo = load_model(os.path.join(RUTA_MODELOS, archivo_mejor))

    gen_tr = datagen_train.flow(X_tr, y_tr, batch_size=BATCH_SIZE, seed=semilla)
    gen_v  = datagen_val.flow(X_v, y_v, batch_size=BATCH_SIZE, shuffle=False)

    historia = modelo.fit(
        gen_tr,
        epochs=EPOCHS,
        validation_data=gen_v,
        callbacks=crear_callbacks(f"{mejor_nombre}_rep{i+1}"),
        verbose=0
    )

    # Evaluamos en el conjunto de prueba de esta particion
    y_prob = modelo.predict(X_te, verbose=0)
    y_pred = (y_prob >= 0.5).astype(int)
    m      = calcular_metricas(y_te, y_pred, y_prob)
    m["Repeticion"] = i + 1
    metricas_fase2.append(m)

    print(f"  Accuracy={m['Accuracy']}  Sensibilidad={m['Sensibilidad']}  "
          f"Especificidad={m['Especificidad']}  F1={m['F1-score']}  AUC={m['AUC']}")

df_fase2 = pd.DataFrame(metricas_fase2)
print(f"\nResumen Fase 2 ({mejor_nombre}):")
print(df_fase2.to_string(index=False))
print(f"\nMedia Accuracy:     {df_fase2['Accuracy'].mean():.4f} "
      f"+/- {df_fase2['Accuracy'].std():.4f}")
print(f"Media Sensibilidad: {df_fase2['Sensibilidad'].mean():.4f} "
      f"+/- {df_fase2['Sensibilidad'].std():.4f}")
print(f"Media F1-score:     {df_fase2['F1-score'].mean():.4f} "
      f"+/- {df_fase2['F1-score'].std():.4f}")
print(f"Media AUC:          {df_fase2['AUC'].mean():.4f} "
      f"+/- {df_fase2['AUC'].std():.4f}")

df_fase2.to_csv(
    os.path.join(RUTA_RESULTADOS, "resultados_fase2.csv"), index=False
)
print("\nResultados fase 2 guardados en RESULTADOS/resultados_fase2.csv")
print("\nPaso 4 completado.")
