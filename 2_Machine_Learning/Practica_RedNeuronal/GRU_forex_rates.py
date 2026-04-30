import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─────────────────────────────────────────────
# 1. PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────
def load_and_clean_data(file_path, target_currency='USD'):
    df = pd.read_csv(file_path)
    df_filtered = df[df['currency'] == target_currency].copy()
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered = df_filtered.sort_values('date')

    # Eliminar duplicados por fecha
    df_filtered = df_filtered.drop_duplicates(subset='date', keep='last').reset_index(drop=True)

    print(f"[INFO] Moneda:   {target_currency}")
    print(f"[INFO] Registros únicos: {len(df_filtered)}")
    print(f"[INFO] Rango:    {df_filtered['date'].min().date()} → {df_filtered['date'].max().date()}")

    if len(df_filtered) < 500:
        raise ValueError(
            f"Solo {len(df_filtered)} registros para {target_currency}. "
            "Se necesitan al menos 500 para usar ventana de 365 días."
        )

    dates  = df_filtered['date'].values
    values = df_filtered['exchange_rate'].values.reshape(-1, 1)

    scaler      = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(values)

    return scaled_data, scaler, dates, df_filtered['exchange_rate'].values


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
        ys.append(data[i + seq_length])
    return (
        torch.FloatTensor(np.array(xs)),
        torch.FloatTensor(np.array(ys)),
    )


# ─────────────────────────────────────────────
# 2. ARQUITECTURA GRU
# ─────────────────────────────────────────────
class ForexGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])   # solo la última salida de la secuencia


# ─────────────────────────────────────────────
# 3. CONFIGURACIÓN
# ─────────────────────────────────────────────
FILE          = 'daily_forex_rates.csv'   # ← cambia la ruta si es necesario
CURRENCY      = 'USD'
SEQ_LENGTH    = 365    # ventana de 1 año (365 días anteriores → predice el día siguiente)
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.2
LEARNING_RATE = 0.001
EPOCHS        = 150
BATCH_SIZE    = 64     # mini-batches para estabilizar el entrenamiento
TRAIN_RATIO   = 0.80

# ─────────────────────────────────────────────
# 4. CARGA Y DIVISIÓN DE DATOS
# ─────────────────────────────────────────────
data_norm, scaler, dates, raw_values = load_and_clean_data(FILE, CURRENCY)

split      = int(len(data_norm) * TRAIN_RATIO)
train_data = data_norm[:split]
test_data  = data_norm[split:]

X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
X_test,  y_test  = create_sequences(test_data,  SEQ_LENGTH)

print(f"[INFO] Train: {len(X_train)} secuencias | Test: {len(X_test)} secuencias")

# Dataset / DataLoader para mini-batches
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ─────────────────────────────────────────────
# 5. MODELO, LOSS Y OPTIMIZADOR
# ─────────────────────────────────────────────
model     = ForexGRU(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Reduce LR si el test loss no mejora en 10 épocas consecutivas
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5
)

# ─────────────────────────────────────────────
# 6. ENTRENAMIENTO
# ─────────────────────────────────────────────
train_losses = []
test_losses  = []
best_test_loss   = float('inf')
best_model_state = None

print("\n[ENTRENAMIENTO]")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # evitar gradient explosion
        optimizer.step()
        epoch_loss += loss.item() * len(X_batch)

    epoch_loss /= len(X_train)
    train_losses.append(epoch_loss)

    # Evaluación en test
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_loss  = criterion(test_preds, y_test).item()
    test_losses.append(test_loss)

    scheduler.step(test_loss)

    # Guardar el mejor modelo
    if test_loss < best_test_loss:
        best_test_loss   = test_loss
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1:3d}/{EPOCHS}] | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss:.6f}")

print(f"\n[INFO] Mejor Test Loss: {best_test_loss:.6f}")

# Restaurar el mejor modelo
model.load_state_dict(best_model_state)

# ─────────────────────────────────────────────
# 7. PREDICCIÓN DEL SIGUIENTE DÍA
# ─────────────────────────────────────────────
model.eval()
with torch.no_grad():
    last_seq      = torch.FloatTensor(data_norm[-SEQ_LENGTH:]).view(1, SEQ_LENGTH, 1)
    pred_norm     = model(last_seq)
    pred_real     = scaler.inverse_transform(pred_norm.numpy())

last_known_rate = raw_values[-1]
predicted_rate  = pred_real[0][0]
change_pct      = (predicted_rate - last_known_rate) / last_known_rate * 100

print(f"\n{'─'*45}")
print(f"  Último valor conocido (USD/EUR): {last_known_rate:.4f}")
print(f"  Predicción siguiente día:        {predicted_rate:.4f}")
print(f"  Cambio estimado:                 {change_pct:+.4f}%")
print(f"{'─'*45}")

# ─────────────────────────────────────────────
# 8. VISUALIZACIÓN
# ─────────────────────────────────────────────
model.eval()
with torch.no_grad():
    test_predictions_norm = model(X_test).numpy()

test_predictions_real = scaler.inverse_transform(test_predictions_norm)
y_test_real           = scaler.inverse_transform(y_test.numpy())

# Fechas correspondientes al conjunto test
test_dates = dates[split + SEQ_LENGTH:]  # alineado con create_sequences

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f"GRU Forex — {CURRENCY}/EUR | Ventana: {SEQ_LENGTH} días", fontsize=14, fontweight='bold')

# --- Panel 1: Real vs Predicho (test set) ---
ax1 = axes[0]
ax1.plot(test_dates, y_test_real,           label='Real',      color='#2196F3', linewidth=1.2)
ax1.plot(test_dates, test_predictions_real, label='Predicción',color='#FF5722', linewidth=1.0, alpha=0.85)
ax1.set_title('Conjunto de Test — Real vs Predicho')
ax1.set_ylabel('Exchange Rate (EUR base)')
ax1.legend()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax1.grid(alpha=0.3)

# --- Panel 2: Curvas de pérdida ---
ax2 = axes[1]
ax2.plot(train_losses, label='Train Loss', color='#4CAF50', linewidth=1.2)
ax2.plot(test_losses,  label='Test Loss',  color='#FF9800', linewidth=1.2)
ax2.set_title('Curvas de Pérdida (MSE)')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('MSE')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('forex_gru_resultado.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[INFO] Gráfico guardado como 'forex_gru_resultado.png'")