import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121, InceptionV3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────
RUTA_RESULTADOS = "/Users/elenasanjuanmunoz/Documents/TFG/RESULTADOS"

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
IMG_SIZE    = (224, 224, 3)  # tamaño de entrada de los modelos
N_SIGNOS    = 6              # número de signos a detectar

# ─────────────────────────────────────────────
# FUNCIÓN PARA CONSTRUIR UN MODELO CON TRANSFER LEARNING
#
# Cada modelo tiene dos partes:
#   1. Base preentrenada: capas que ya saben detectar bordes, texturas, formas
#   2. Cabeza personalizada: capas nuevas que aprenden a detectar RD y sus signos
# ─────────────────────────────────────────────
def construir_modelo(arquitectura, n_salidas, activacion_salida, nombre):
    """
    arquitectura        : ResNet50, DenseNet121 o InceptionV3
    n_salidas           : 1 para clasificación binaria, 6 para multietiqueta
    activacion_salida   : 'sigmoid' para ambos casos
    nombre              : nombre identificativo del modelo
    """

    # Cargamos la base preentrenada con ImageNet
    # include_top=False, quitamos la capa final originaponemos la nuestra adaptada a nuestro problema
    base = arquitectura(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE
    )

    # Fase 1: congelamos todas las capas de la base
    # Solo entrenaremos las capas nuevas que añadimos
    base.trainable = False

    # Añadimos nuestra cabeza personalizada encima de la base
    x = base.output
    x = GlobalAveragePooling2D()(x)   # reduce el mapa de características a un vector
    x = Dense(256, activation='relu')(x)  # capa densa con 256 neuronas
    x = Dropout(0.4)(x)               # dropout para evitar sobreajuste (40% neuronas apagadas)
    x = Dense(128, activation='relu')(x)  # segunda capa densa con 128 neuronas
    x = Dropout(0.3)(x)               # segundo dropout

    # Capa de salida
    # sigmoid: cada salida es un valor entre 0 y 1 (probabilidad)
    salida = Dense(n_salidas, activation=activacion_salida)(x)

    modelo = Model(inputs=base.input, outputs=salida, name=nombre)

    return modelo, base


# ─────────────────────────────────────────────
# CONSTRUIR LOS TRES MODELOS PARA DETECCIÓN DE SIGNOS
# (tarea multietiqueta — 6 signos)
# ─────────────────────────────────────────────
print("Construyendo modelos para detección de signos (multietiqueta)...\n")

resnet_signos, resnet_base = construir_modelo(
    ResNet50, N_SIGNOS, 'sigmoid', 'ResNet50_signos'
)

densenet_signos, densenet_base = construir_modelo(
    DenseNet121, N_SIGNOS, 'sigmoid', 'DenseNet121_signos'
)

inception_signos, inception_base = construir_modelo(
    InceptionV3, N_SIGNOS, 'sigmoid', 'InceptionV3_signos'
)

# ─────────────────────────────────────────────
# COMPILAR LOS MODELOS
# Compilar = definir cómo van a aprender
#   - optimizer: Adam ajusta los pesos durante el entrenamiento
#   - loss: binary_crossentropy para multietiqueta (cada signo es independiente)
#   - metrics: accuracy y AUC para evaluar el rendimiento
# ─────────────────────────────────────────────
def compilar_modelo(modelo):
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='sensibilidad'),
            tf.keras.metrics.Precision(name='precision')
        ]
    )

compilar_modelo(resnet_signos)
compilar_modelo(densenet_signos)
compilar_modelo(inception_signos)

# ─────────────────────────────────────────────
# RESUMEN DE LOS MODELOS
# ─────────────────────────────────────────────
modelos = {
    'ResNet50':     resnet_signos,
    'DenseNet121':  densenet_signos,
    'InceptionV3':  inception_signos
}

print("Resumen de modelos construidos:\n")
for nombre, modelo in modelos.items():
    total_params     = modelo.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in modelo.trainable_weights])
    frozen_params    = total_params - trainable_params
    print(f"  {nombre}:")
    print(f"    Parámetros totales:      {total_params:,}")
    print(f"    Parámetros entrenables:  {trainable_params:,}  (capas nuevas)")
    print(f"    Parámetros congelados:   {frozen_params:,}  (base preentrenada)")
    print()

# ─────────────────────────────────────────────
# GUARDAR LOS MODELOS (arquitectura lista para entrenar)
# ─────────────────────────────────────────────
RUTA_MODELOS = os.path.join(RUTA_RESULTADOS, "modelos")
os.makedirs(RUTA_MODELOS, exist_ok=True)

resnet_signos.save(os.path.join(RUTA_MODELOS,   "resnet50_signos.keras"))
densenet_signos.save(os.path.join(RUTA_MODELOS, "densenet121_signos.keras"))
inception_signos.save(os.path.join(RUTA_MODELOS,"inceptionv3_signos.keras"))

print("Modelos guardados en TFG/RESULTADOS/modelos/")
print("\nPaso 3 completado. Listos para entrenar.")
