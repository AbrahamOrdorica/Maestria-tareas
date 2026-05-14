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

GRU_CONFIG_NAME = "base"
RUN_ALL_CONFIGS = False
TRAIN_RATIO = 0.8
PRINT_ROWS = 20

GRU_CONFIGS = {
    "base": {"gru1": 64, "gru2": 32, "dropout": 0.2, "dense": 32, "lr": 0.001},
    "small_gru": {"gru1": 32, "gru2": 16, "dropout": 0.2, "dense": 32, "lr": 0.001},
    "large_gru": {"gru1": 128, "gru2": 64, "dropout": 0.2, "dense": 64, "lr": 0.001},
    "dropout_low": {"gru1": 64, "gru2": 32, "dropout": 0.1, "dense": 32, "lr": 0.001},
    "dropout_high": {"gru1": 64, "gru2": 32, "dropout": 0.4, "dense": 32, "lr": 0.001},
    "balanced": {"gru1": 96, "gru2": 48, "dropout": 0.3, "dense": 48, "lr": 0.001},
}


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


def dividir_temporalmente(X, y, fechas, train_ratio=TRAIN_RATIO):
    train_size = int(len(X) * train_ratio)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:], fechas[:train_size], fechas[train_size:]


def construir_modelo(input_shape, config_name=GRU_CONFIG_NAME):
    config = GRU_CONFIGS[config_name]
    model = Sequential([
        Input(shape=input_shape),
        GRU(config["gru1"], return_sequences=True),
        Dropout(config["dropout"]),
        GRU(config["gru2"]),
        Dropout(config["dropout"]),
        Dense(config["dense"], activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=config["lr"]), loss="mse", metrics=["mae"])
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


def guardar_errores(fechas_test, y_real, y_pred, output_dir, print_rows=PRINT_ROWS):
    resultados = pd.DataFrame({
        "fecha": pd.to_datetime(fechas_test),
        "valor_real": y_real.ravel(),
        "valor_predicho": y_pred.ravel(),
    })
    resultados["error"] = resultados["valor_real"] - resultados["valor_predicho"]
    resultados["error_absoluto"] = resultados["error"].abs()

    path = output_dir / "forex_real_predicho_error.csv"
    resultados.to_csv(path, index=False)
    print(f"Tabla real/predicho/error guardada en: {path}")

    if print_rows > 0:
        print("\nMUESTRA REAL VS PREDICHO")
        print(resultados.head(print_rows).to_string(index=False))

    return resultados


def ejecutar_forex(ruta_csv, base_currency, currency, window_size, epochs, batch_size, output_dir, config_name=GRU_CONFIG_NAME, train_ratio=TRAIN_RATIO, print_rows=PRINT_ROWS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuracion: {config_name} -> {GRU_CONFIGS[config_name]}")
    df, target_col = cargar_forex_csv(ruta_csv, base_currency, currency)
    graficar_serie_completa(df, target_col, output_dir)

    X, y, fechas, scaler = crear_secuencias(df, target_col, window_size)
    X_train, X_test, y_train, y_test, _, fechas_test = dividir_temporalmente(X, y, fechas, train_ratio=train_ratio)

    model = construir_modelo(input_shape=(X_train.shape[1], X_train.shape[2]), config_name=config_name)
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
    print(f"Configuracion: {config_name}")
    print(f"Paridad objetivo: {target_col}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    graficar_historial(history, output_dir)
    graficar_predicciones(fechas_test, y_test_real, y_pred_real, output_dir, target_col)
    guardar_errores(fechas_test, y_test_real, y_pred_real, output_dir, print_rows)

    model_path = output_dir / "modelo_gru_forex.keras"
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")
    return {"config": config_name, "mae": mae, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser(description="GRU para prediccion numerica con Forex")
    parser.add_argument("--forex_csv", type=str, default=str(DEFAULT_FOREX))
    parser.add_argument("--base_currency", type=str, default="EUR")
    parser.add_argument("--currency", type=str, default="MXN")
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", choices=list(GRU_CONFIGS.keys()), default=GRU_CONFIG_NAME)
    parser.add_argument("--run_all_configs", action="store_true", default=RUN_ALL_CONFIGS)
    parser.add_argument("--print_rows", type=int, default=PRINT_ROWS)
    args = parser.parse_args()

    configs = list(GRU_CONFIGS.keys()) if args.run_all_configs else [args.config]
    resumen = []
    for config_name in configs:
        run_output = Path(args.output_dir) / config_name if args.run_all_configs else Path(args.output_dir)
        resumen.append(ejecutar_forex(
            ruta_csv=args.forex_csv,
            base_currency=args.base_currency,
            currency=args.currency,
            window_size=args.window_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=run_output,
            config_name=config_name,
            train_ratio=args.train_ratio,
            print_rows=args.print_rows,
        ))

    if len(resumen) > 1:
        resumen_df = pd.DataFrame(resumen).sort_values("rmse")
        resumen_path = Path(args.output_dir) / "forex_resumen_configuraciones.csv"
        resumen_path.parent.mkdir(parents=True, exist_ok=True)
        resumen_df.to_csv(resumen_path, index=False)
        print("\nRESUMEN DE CONFIGURACIONES")
        print(resumen_df.to_string(index=False))
        print(f"Resumen guardado en: {resumen_path}")


if __name__ == "__main__":
    main()
