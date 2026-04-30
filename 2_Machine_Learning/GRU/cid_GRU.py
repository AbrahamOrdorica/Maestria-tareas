import argparse
import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pdfplumber
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "Cantar del mio Cid.pdf"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "cid"


def configurar_gpu(usar_mixed_precision=False):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detectada por TensorFlow:")
        for gpu in gpus:
            print(f"- {gpu.name}")
        if usar_mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision activado.")
    else:
        print("TensorFlow no detecto GPU. El entrenamiento usara CPU.")
    return gpus


def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"[^\w .,;:!?'\-]", "", texto)
    return texto.strip()


def extraer_texto_pdf(ruta_pdf, max_chars=None):
    ruta_pdf = Path(ruta_pdf)
    if not ruta_pdf.exists():
        raise FileNotFoundError(f"No se encontro el PDF: {ruta_pdf}")

    texto = ""
    print("Extrayendo texto del PDF...")
    with pdfplumber.open(ruta_pdf) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                texto += page_text + "\n"

    texto = limpiar_texto(texto)
    if max_chars is not None and max_chars > 0:
        texto = texto[:max_chars]

    if len(texto) < 500:
        raise ValueError("El texto extraido es demasiado corto. Revisa si el PDF permite extraer texto.")

    print(f"Caracteres extraidos: {len(texto)}")
    return texto


def crear_dataset_caracteres(texto, seq_length=80, step=3):
    chars = sorted(list(set(texto)))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    X, y = [], []
    for i in range(0, len(texto) - seq_length, step):
        secuencia = texto[i:i + seq_length]
        siguiente_char = texto[i + seq_length]
        X.append([char_to_idx[c] for c in secuencia])
        y.append(char_to_idx[siguiente_char])

    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    if len(X) == 0:
        raise ValueError("No hay suficientes caracteres. Reduce --seq_length.")

    print(f"Vocabulario de caracteres: {len(chars)}")
    print(f"Secuencias generadas: {len(X)}")
    return X, y, chars, char_to_idx, idx_to_char


def crear_tf_dataset(X, y, batch_size, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(X), 10000), seed=SEED)
    return dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)


def construir_modelo(vocab_size, seq_length, embedding_dim=96, gru_units=192, dropout=0.2):
    model = Sequential([
        Input(shape=(seq_length,)),
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        GRU(gru_units, return_sequences=True, reset_after=True),
        Dropout(dropout),
        GRU(gru_units, reset_after=True),
        Dropout(dropout),
        Dense(vocab_size, activation="softmax", dtype="float32"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def muestrear_prediccion(predicciones, temperatura=0.45, top_k=10):
    predicciones = np.asarray(predicciones).astype("float64")

    if top_k is not None and top_k > 0 and top_k < len(predicciones):
        indices_top = np.argpartition(predicciones, -top_k)[-top_k:]
        filtradas = np.zeros_like(predicciones)
        filtradas[indices_top] = predicciones[indices_top]
        predicciones = filtradas

    predicciones = np.log(predicciones + 1e-8) / temperatura
    exp_preds = np.exp(predicciones)
    predicciones = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(predicciones), p=predicciones)


def generar_texto(model, semilla, char_to_idx, idx_to_char, seq_length, longitud, temperatura, top_k):
    texto_generado = semilla.lower()
    if len(texto_generado) < seq_length:
        texto_generado = texto_generado.rjust(seq_length)

    ventana = texto_generado[-seq_length:]
    for _ in range(longitud):
        x = np.array([[char_to_idx.get(c, 0) for c in ventana]], dtype=np.int32)
        predicciones = model.predict(x, verbose=0)[0]
        siguiente_idx = muestrear_prediccion(predicciones, temperatura=temperatura, top_k=top_k)
        siguiente_char = idx_to_char[siguiente_idx]
        texto_generado += siguiente_char
        ventana = texto_generado[-seq_length:]

    return texto_generado


def graficar_historial(history, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validacion")
    plt.title("Cantar del Mio Cid - perdida por epoca")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "cid_loss.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Grafica guardada en: {path}")


def ejecutar_cid(
    ruta_pdf,
    seq_length,
    step,
    epochs,
    batch_size,
    output_dir,
    temperatura,
    longitud_generada,
    top_k=10,
    max_chars=None,
    patience=5,
    usar_mixed_precision=False,
):
    configurar_gpu(usar_mixed_precision=usar_mixed_precision)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texto = extraer_texto_pdf(ruta_pdf, max_chars=max_chars)
    X, y, chars, char_to_idx, idx_to_char = crear_dataset_caracteres(texto, seq_length, step)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    train_ds = crear_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_ds = crear_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    model = construir_modelo(vocab_size=len(chars), seq_length=seq_length)
    model.summary()

    checkpoint_path = output_dir / "mejor_modelo_gru_cid.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(test_ds, verbose=1)
    print("\nRESULTADOS CANTAR DEL MIO CID")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")

    graficar_historial(history, output_dir)

    texto_generado = generar_texto(
        model,
        semilla=texto[:seq_length],
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        seq_length=seq_length,
        longitud=longitud_generada,
        temperatura=temperatura,
        top_k=top_k,
    )

    generated_path = output_dir / "texto_generado_cid.txt"
    generated_path.write_text(texto_generado, encoding="utf-8")
    print(f"Texto generado guardado en: {generated_path}")

    vocab_path = output_dir / "cid_vocabulario.json"
    vocab_path.write_text(
        json.dumps({"chars": chars, "char_to_idx": char_to_idx}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Vocabulario guardado en: {vocab_path}")

    model_path = output_dir / "modelo_gru_cid.keras"
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="GRU para prediccion de caracteres con Cantar del Mio Cid")
    parser.add_argument("--pdf", type=str, default=str(DEFAULT_PDF))
    parser.add_argument("--seq_length", type=int, default=80)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--temperatura", type=float, default=0.45)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--longitud_generada", type=int, default=700)
    parser.add_argument("--max_chars", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--mixed_precision", action="store_true")
    args = parser.parse_args()

    ejecutar_cid(
        ruta_pdf=args.pdf,
        seq_length=args.seq_length,
        step=args.step,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        temperatura=args.temperatura,
        longitud_generada=args.longitud_generada,
        top_k=args.top_k,
        max_chars=args.max_chars,
        patience=args.patience,
        usar_mixed_precision=args.mixed_precision,
    )


if __name__ == "__main__":
    main()