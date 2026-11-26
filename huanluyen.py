import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Đọc file CSV đã tải
df = pd.read_csv("hanoi_weather.csv")
temps = df["temperature_2m"].ffill().bfill().values.astype(np.float32)

def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

class WeatherModel(nn.Module):
    def __init__(self, model_type="LSTM", input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[model_type]
        self.rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# 2. Chia Train/Test (80/20) và chuẩn hóa
SEQ_LEN = 24
split_idx = int(len(temps) * 0.8)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(temps[:split_idx].reshape(-1, 1))
test_scaled = scaler.transform(temps[split_idx - SEQ_LEN :].reshape(-1, 1))
joblib.dump(scaler, "scaler.pkl")

X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
X_test, y_test = create_sequences(test_scaled, SEQ_LEN)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
    batch_size=64,
    shuffle=True,
)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_real = scaler.inverse_transform(y_test).flatten()

# 3. Huấn luyện từng mô hình
results = {}
print(f"Đang chạy trên: {device}")
print(f"Số mẫu Train: {len(X_train)} giờ | Số mẫu Test: {len(X_test)} giờ")
print(f"{'Mô hình':<8} | {'MAE (°C)':<10} | {'RMSE (°C)':<10} | {'Thời gian (s)':<12}")
print("-" * 50)

for m_type in ["RNN", "LSTM", "GRU"]:
    model = WeatherModel(model_type=m_type).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    t0 = time.time()
    model.train()
    for epoch in range(25):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    t_sec = time.time() - t0

    # Lưu trọng số mô hình
    torch.save(model.state_dict(), f"{m_type}_weather.pth")

    # Đánh giá trên toàn bộ tập Test
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t.to(device)).cpu().numpy()
    preds_real = scaler.inverse_transform(preds_scaled).flatten()

    mae = mean_absolute_error(y_test_real, preds_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
    results[m_type] = preds_real
    print(f"{m_type:<8} | {mae:<10.4f} | {rmse:<10.4f} | {t_sec:<12.2f}")

# 4. Vẽ biểu đồ: Trên = Toàn bộ tập Test, Dưới = Phóng to 7 ngày đầu
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

# Biểu đồ 1: Toàn bộ tập Test
ax1.plot(y_test_real, label="Thực tế", color="black", linewidth=1.2)
ax1.plot(results["RNN"], label="RNN", linestyle="--", alpha=0.75, linewidth=1)
ax1.plot(results["LSTM"], label="LSTM", linestyle="-.", alpha=0.8, linewidth=1)
ax1.plot(results["GRU"], label="GRU", linestyle=":", alpha=0.9, linewidth=1.2)
ax1.set_title(f"So sánh dự báo nhiệt độ trên TOÀN BỘ tập Test ({len(y_test_real)} giờ)")
ax1.set_xlabel("Thời gian (Giờ)")
ax1.set_ylabel("Nhiệt độ (°C)")
ax1.legend()
ax1.grid(alpha=0.3)

# Biểu đồ 2: Phóng to 168 giờ đầu (7 ngày) để soi chi tiết
ax2.plot(y_test_real[:168], label="Thực tế", color="black", linewidth=2)
ax2.plot(results["RNN"][:168], label="RNN", linestyle="--", alpha=0.8)
ax2.plot(results["LSTM"][:168], label="LSTM", linestyle="-.", alpha=0.85)
ax2.plot(results["GRU"][:168], label="GRU", linestyle=":", linewidth=2)
ax2.set_title("Phóng to 7 ngày đầu của tập Test (168 giờ)")
ax2.set_xlabel("Thời gian (Giờ)")
ax2.set_ylabel("Nhiệt độ (°C)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("weather_comparison_full.png", dpi=150)
plt.show()
print("Đã lưu ảnh: weather_comparison_full.png")