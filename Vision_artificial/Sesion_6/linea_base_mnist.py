"""
Línea base mínima: clasificador de dígitos MNIST con una CNN pequeña.

Keras 3 con backend PyTorch. El código prioriza la legibilidad sobre la
eficiencia: cada etapa está separada y comentada.

Es solo el punto de partida de la actividad de la Sesión 6 (CIFAR-100).
"""

# =============================================================================
# 1. Backend
# =============================================================================
# Keras 3 puede ejecutarse sobre TensorFlow, JAX o PyTorch.
# Elegimos PyTorch. La variable debe definirse ANTES de importar keras.
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers

# Semilla fija para que los resultados se puedan repetir.
keras.utils.set_random_seed(42)


# =============================================================================
# 2. Datos
# =============================================================================
# MNIST: 60 000 imágenes de entrenamiento y 10 000 de prueba,
# de 28x28 píxeles en escala de grises, con etiquetas de 0 a 9.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Entrenamiento:", x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print("Prueba:       ", x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)


# =============================================================================
# 3. Preprocesamiento
# =============================================================================
# a) Escalar los píxeles de 0-255 a 0-1.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# b) Agregar el eje de canales: Keras espera (alto, ancho, canales).
#    MNIST tiene un solo canal (escala de grises).
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# =============================================================================
# 4. Modelo
# =============================================================================
# Dos bloques convolución + max pooling, y una capa densa de salida.
# Los comentarios indican la forma del volumen después de cada capa.
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),                       # 28 x 28 x 1

    layers.Conv2D(16, kernel_size=3, activation="relu"),  # 26 x 26 x 16
    layers.MaxPooling2D(pool_size=2),                     # 13 x 13 x 16

    layers.Conv2D(32, kernel_size=3, activation="relu"),  # 11 x 11 x 32
    layers.MaxPooling2D(pool_size=2),                     #  5 x  5 x 32

    layers.Flatten(),                                     # 800
    layers.Dense(10, activation="softmax"),               # 10 probabilidades
])

# Tabla con la forma de salida y los parámetros de cada capa.
model.summary()


# =============================================================================
# 5. Compilación
# =============================================================================
# - optimizer: cómo se actualizan los pesos (Adam).
# - loss: entropía cruzada para etiquetas enteras (0, 1, ..., 9).
# - metrics: qué se reporta durante el entrenamiento.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# =============================================================================
# 6. Entrenamiento
# =============================================================================
# - batch_size: imágenes por paso de gradiente.
# - epochs: pasadas completas sobre los datos de entrenamiento.
# - validation_split: fracción del entrenamiento que se aparta para validar.
#   El conjunto de prueba NO se usa aquí.
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=3,
    validation_split=0.1,
)


# =============================================================================
# 7. Evaluación
# =============================================================================
# Se usa el conjunto de prueba una sola vez, al final.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Accuracy en el conjunto de prueba: {test_accuracy:.4f}")
