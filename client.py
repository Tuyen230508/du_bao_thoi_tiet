import tkinter as tk
from tkinter import ttk, messagebox
import requests
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class WeatherModel(nn.Module):
    def __init__(self, model_type="LSTM", input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[model_type]
        self.rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class WeatherClientApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dự báo nhiệt độ ngày mai - Hà Nội")
        self.root.geometry("760x540")
        self.root.resizable(False, False)

        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(fill="x")

        ttk.Label(control_frame, text="Chọn mô hình:").pack(side="left", padx=5)
        self.model_var = tk.StringVar(value="GRU")
        ttk.Combobox(
            control_frame,
            textvariable=self.model_var,
            values=["RNN", "LSTM", "GRU"],
            state="readonly",
            width=10,
        ).pack(side="left", padx=5)

        ttk.Button(control_frame, text="Dự báo ngày hôm sau", command=self.run_predict).pack(side="left", padx=10)

        self.result_label = ttk.Label(root, text="Chọn mô hình và bấm Dự báo ngày hôm sau", font=("Segoe UI", 12, "bold"))
        self.result_label.pack(pady=8)

        self.fig, self.ax = plt.subplots(figsize=(7, 3.8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

    def get_last_24h_temps(self):
        try:
            url = (
                "https://api.open-meteo.com/v1/forecast"
                "?latitude=21.0285&longitude=105.8542"
                "&hourly=temperature_2m&past_days=1&forecast_days=1"
            )
            temps = requests.get(url, timeout=8).json()["hourly"]["temperature_2m"][:24]
            return np.array(temps, dtype=np.float32), "Trực tuyến"
        except Exception:
            df = pd.read_csv("hanoi_weather.csv")
            temps = df["temperature_2m"].dropna().values[-24:].astype(np.float32)
            return temps, "Offline"

    def run_predict(self):
        model_type = self.model_var.get()
        try:
            temps, source = self.get_last_24h_temps()
            scaler = joblib.load("scaler.pkl")

            model = WeatherModel(model_type=model_type)
            model.load_state_dict(torch.load(f"{model_type}_weather.pth", map_location="cpu"))
            model.eval()

            # Dự báo cuốn chiếu 24 giờ của ngày hôm sau
            current_seq = scaler.transform(temps.reshape(-1, 1)).flatten().tolist()
            preds_scaled = []

            with torch.no_grad():
                for _ in range(24):
                    x_in = torch.tensor(current_seq[-24:], dtype=torch.float32).reshape(1, 24, 1)
                    next_val = model(x_in).item()
                    preds_scaled.append(next_val)
                    current_seq.append(next_val)

            next_day_temps = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

            t_avg = np.mean(next_day_temps)
            t_min = np.min(next_day_temps)
            t_max = np.max(next_day_temps)

            self.result_label.config(
                text=f"Ngày mai ({model_type}): TB {t_avg:.1f}°C  |  Thấp nhất {t_min:.1f}°C  |  Cao nhất {t_max:.1f}°C  ({source})"
            )

            # Chỉ vẽ biểu đồ 24 giờ của ngày hôm sau
            self.ax.clear()
            hours = list(range(1, 25))
            self.ax.plot(hours, next_day_temps, marker="o", color="crimson", linewidth=2, label=f"Dự báo ngày mai ({model_type})")
            self.ax.axhline(t_avg, color="gray", linestyle="--", alpha=0.7, label=f"Trung bình ({t_avg:.1f}°C)")
            self.ax.set_title(f"Dự báo nhiệt độ 24 giờ ngày hôm sau ({model_type})")
            self.ax.set_xlabel("Giờ trong ngày mai (1h - 24h)")
            self.ax.set_ylabel("Nhiệt độ (°C)")
            self.ax.set_xticks(range(0, 25, 2))
            self.ax.legend()
            self.ax.grid(alpha=0.3)
            self.canvas.draw()

        except FileNotFoundError as e:
            messagebox.showerror("Thiếu file", f"Chưa có file mô hình hoặc scaler.\n({e})")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherClientApp(root)
    root.mainloop()