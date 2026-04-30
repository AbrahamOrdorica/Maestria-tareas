import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FOREX = BASE_DIR / "Forex" / "daily_forex_rates.csv"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "forex"


def cargar_forex_csv(ruta_csv=DEFAULT_FOREX, base_currency="EUR", currency="MXN"):
    ruta_csv = Path(ruta_csv)
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontro el CSV de Forex: {ruta_csv}")

    print(f"Cargando Forex desde: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    required = {"currency", "base_currency", "exchange_rate", "date"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Al CSV le faltan columnas: {sorted(missing)}")

    serie = df[
        (df["base_currency"].astype(str).str.upper() == base_currency.upper())
        & (df["currency"].astype(str).str.upper() == currency.upper())
    ].copy()

    if serie.empty:
        raise ValueError(f"No se encontro la paridad {currency.upper()}/{base_currency.upper()} en el CSV.")

    serie["date"] = pd.to_datetime(serie["date"])
    serie = serie.sort_values("date")
    serie = serie.set_index("date")[["exchange_rate"]]
    target_col = f"{currency.upper()}/{base_currency.upper()}"
    serie = serie.rename(columns={"exchange_rate": target_col})
    serie = serie.replace([np.inf, -np.inf], np.nan).interpolate().ffill().bfill()

    print("Datos Forex cargados correctamente.")
    print(serie.head())
    return serie, target_col


def crear_secuencias(df, target_col, window_size=30):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[target_col]].values)

    X, y, fechas_y = [], [], []
    for i in range(window_size, len(df)):
        X.append(scaled[i - window_size:i, :])
        y.append(scaled[i, 0])
        fechas_y.append(df.index[i])

    X = np.array(X)
    y = np.array(y)
    fechas_y = np.array(fechas_y)

    if len(X) == 0:
        raise ValueError("No hay suficientes datos. Reduce --window_size.")

    return X, y, fechas_y, scaler


def dividir_temporalmente(X, y, fechas, train_ratio=0.8):
    train_size = int(len(X) * train_ratio)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:], fechas[:train_size], fechas[train_size:]


def construir_modelo(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def graficar_serie_completa(df, target_col, output_dir):
    serie = df[target_col]

    plt.figure(figsize=(15, 4))
    plt.plot(df.index, serie, linewidth=0.8, color="#185FA5")
    plt.title(f"Forex - {target_col}")
    plt.xlabel("Fecha")
    plt.ylabel(target_col)
    plt.tight_layout()
    path = output_dir / "forex_patron_completo.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Grafica guardada en: {path}")
    print(f"Minima: {serie.min():.4f}")
    print(f"Maxima: {serie.max():.4f}")
    print(f"Promedio: {serie.mean():.4f}")


def graficar_historial(history, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validacion")
    plt.title("Forex - perdida por epoca")
    plt.xlabel("Epoca")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "forex_loss.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Grafica guardada en: {path}")


def graficar_predicciones(fechas_test, y_real, y_pred, output_dir, target_col):
    plt.figure(figsize=(15, 5))
    plt.plot(fechas_test, y_real, label="Real")
    plt.plot(fechas_test, y_pred, label="Prediccion")
    plt.title(f"{target_col} real vs predicho")
    plt.xlabel("Fecha de prueba")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    path = output_dir / "forex_predicciones.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Grafica guardada en: {path}")


def ejecutar_forex(ruta_csv, base_currency, currency, window_size, epochs, batch_size, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, target_col = cargar_forex_csv(ruta_csv, base_currency, currency)
    graficar_serie_completa(df, target_col, output_dir)

    X, y, fechas, scaler = crear_secuencias(df, target_col, window_size)
    X_train, X_test, y_train, y_test, _, fechas_test = dividir_temporalmente(X, y, fechas)

    model = construir_modelo(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = scaler.inverse_transform(y_pred_scaled)

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    print("\nRESULTADOS FOREX")
    print(f"Paridad objetivo: {target_col}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    graficar_historial(history, output_dir)
    graficar_predicciones(fechas_test, y_test_real, y_pred_real, output_dir, target_col)

    model_path = output_dir / "modelo_gru_forex.keras"
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="GRU para prediccion numerica con Forex")
    parser.add_argument("--forex_csv", type=str, default=str(DEFAULT_FOREX))
    parser.add_argument("--base_currency", type=str, default="EUR")
    parser.add_argument("--currency", type=str, default="MXN")
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    ejecutar_forex(
        ruta_csv=args.forex_csv,
        base_currency=args.base_currency,
        currency=args.currency,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()