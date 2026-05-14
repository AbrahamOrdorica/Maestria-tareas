import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
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
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "nasa"

NASA_VARIABLES = {
    "T2M": "Temperature at 2 Meters",
    "WS2M": "Wind Speed at 2 Meters",
}

NASA_UNITS = {
    "T2M": "C",
    "WS2M": "m/s",
}

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


def nombre_variable(col):
    return f"{col} ({NASA_VARIABLES.get(col, col)})"


def descargar_nasa_power(lat=20.5888, lon=-100.3899, start="20190101", end="20241231"):
    parameters = ",".join(NASA_VARIABLES.keys())
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={parameters}"
        "&community=RE"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start}"
        f"&end={end}"
        "&format=JSON"
    )

    print("Descargando datos de NASA POWER...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    data = response.json()["properties"]["parameter"]
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.sort_index()
    df = df.replace([-999, -999.0], np.nan).interpolate().ffill().bfill()

    print("Datos NASA descargados correctamente.")
    print("Variables descargadas y utilizadas como entrada del modelo:")
    for col in df.columns:
        print(f"- {nombre_variable(col)} ({NASA_UNITS.get(col, 'sin unidad')})")
    print(df.head())
    return df


def crear_secuencias(df, target_col, window_size=30):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(df.values)
    target_scaled = target_scaler.fit_transform(df[[target_col]].values)

    X, y, fechas_y = [], [], []
    for i in range(window_size, len(df)):
        X.append(features_scaled[i - window_size:i, :])
        y.append(target_scaled[i, 0])
        fechas_y.append(df.index[i])

    X = np.array(X)
    y = np.array(y)
    fechas_y = np.array(fechas_y)

    if len(X) == 0:
        raise ValueError("No hay suficientes datos. Reduce --window_size o amplia el rango de fechas.")

    return X, y, fechas_y, target_scaler


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
    etiqueta = nombre_variable(target_col)
    unidad = NASA_UNITS.get(target_col, "")
    serie = df[target_col]

    plt.figure(figsize=(15, 4))
    plt.plot(df.index, serie, linewidth=0.8, color="#185FA5")
    plt.title(f"NASA POWER - {etiqueta}")
    plt.xlabel("Fecha")
    plt.ylabel(f"{target_col} ({unidad})" if unidad else target_col)
    plt.tight_layout()
    path = output_dir / f"nasa_{target_col}_patron_completo.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Grafica guardada en: {path}")
    print(f"Minima {target_col}: {serie.min():.2f} {unidad}")
    print(f"Maxima {target_col}: {serie.max():.2f} {unidad}")
    print(f"Promedio {target_col}: {serie.mean():.2f} {unidad}")


def graficar_historial(history, output_dir, target_col):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validacion")
    plt.title(f"Perdida por epoca - {nombre_variable(target_col)}")
    plt.xlabel("Epoca")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    path = output_dir / f"nasa_{target_col}_loss.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Grafica guardada en: {path}")


def graficar_predicciones(fechas_test, y_real, y_pred, output_dir, target_col):
    etiqueta = nombre_variable(target_col)
    unidad = NASA_UNITS.get(target_col, "")

    plt.figure(figsize=(15, 5))
    plt.plot(fechas_test, y_real, label="Real")
    plt.plot(fechas_test, y_pred, label="Prediccion")
    plt.title(f"{etiqueta} real vs predicho")
    plt.xlabel("Fecha de prueba")
    plt.ylabel(f"{target_col} ({unidad})" if unidad else target_col)
    plt.legend()
    plt.tight_layout()
    path = output_dir / f"nasa_{target_col}_predicciones.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Grafica guardada en: {path}")


def guardar_errores(fechas_test, y_real, y_pred, output_dir, target_col, print_rows=PRINT_ROWS):
    resultados = pd.DataFrame({
        "fecha": pd.to_datetime(fechas_test),
        "valor_real": y_real.ravel(),
        "valor_predicho": y_pred.ravel(),
    })
    resultados["error"] = resultados["valor_real"] - resultados["valor_predicho"]
    resultados["error_absoluto"] = resultados["error"].abs()

    path = output_dir / f"nasa_{target_col}_real_predicho_error.csv"
    resultados.to_csv(path, index=False)
    print(f"Tabla real/predicho/error guardada en: {path}")

    if print_rows > 0:
        print(f"\nMUESTRA REAL VS PREDICHO - {target_col}")
        print(resultados.head(print_rows).to_string(index=False))

    return resultados



def entrenar_variable(df, target_col, window_size, epochs, batch_size, output_dir, config_name=GRU_CONFIG_NAME, train_ratio=TRAIN_RATIO, print_rows=PRINT_ROWS):
    print("\n" + "=" * 70)
    print(f"Entrenando: {nombre_variable(target_col)}")
    print(f"Configuracion: {config_name} -> {GRU_CONFIGS[config_name]}")
    print("=" * 70)

    graficar_serie_completa(df, target_col, output_dir)

    X, y, fechas, target_scaler = crear_secuencias(df, target_col=target_col, window_size=window_size)
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
    y_test_real = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = target_scaler.inverse_transform(y_pred_scaled)

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    print(f"\nRESULTADOS - {nombre_variable(target_col)}")
    print(f"Configuracion: {config_name}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    graficar_historial(history, output_dir, target_col)
    graficar_predicciones(fechas_test, y_test_real, y_pred_real, output_dir, target_col)
    guardar_errores(fechas_test, y_test_real, y_pred_real, output_dir, target_col, print_rows)

    model_path = output_dir / f"modelo_gru_nasa_{target_col}.keras"
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")
    return {"target_col": target_col, "config": config_name, "mae": mae, "rmse": rmse}

def ejecutar_nasa(lat, lon, start, end, target_cols, window_size, epochs, batch_size, output_dir, config_name=GRU_CONFIG_NAME, train_ratio=TRAIN_RATIO, print_rows=PRINT_ROWS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = descargar_nasa_power(lat=lat, lon=lon, start=start, end=end)
    resultados = []

    for target_col in target_cols:
        if target_col not in df.columns:
            raise ValueError(f"{target_col} no esta en NASA POWER. Opciones validas: {list(df.columns)}")
        resultados.append(entrenar_variable(df, target_col, window_size, epochs, batch_size, output_dir, config_name, train_ratio, print_rows))

    return resultados

def main():
    parser = argparse.ArgumentParser(description="GRU para prediccion numerica con NASA POWER")
    parser.add_argument("--lat", type=float, default=20.5888)
    parser.add_argument("--lon", type=float, default=-100.3899)
    parser.add_argument("--start", type=str, default="20190101")
    parser.add_argument("--end", type=str, default="20241231")
    parser.add_argument("--target_cols", nargs="+", default=["T2M", "WS2M"], choices=list(NASA_VARIABLES.keys()))
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
        resumen.extend(ejecutar_nasa(
            lat=args.lat,
            lon=args.lon,
            start=args.start,
            end=args.end,
            target_cols=args.target_cols,
            window_size=args.window_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=run_output,
            config_name=config_name,
            train_ratio=args.train_ratio,
            print_rows=args.print_rows,
        ))

    if len(resumen) > len(args.target_cols):
        resumen_df = pd.DataFrame(resumen).sort_values(["target_col", "rmse"])
        resumen_path = Path(args.output_dir) / "nasa_resumen_configuraciones.csv"
        resumen_path.parent.mkdir(parents=True, exist_ok=True)
        resumen_df.to_csv(resumen_path, index=False)
        print("\nRESUMEN DE CONFIGURACIONES")
        print(resumen_df.to_string(index=False))
        print(f"Resumen guardado en: {resumen_path}")


if __name__ == "__main__":
    main()
