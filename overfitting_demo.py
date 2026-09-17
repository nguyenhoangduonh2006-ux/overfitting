# -*- coding: utf-8 -*-
"""
DEMO: OVERFITTING -> EARLY STOPPING (theo đúng slide "Early Stopping")
Dữ liệu: du_lieu_gia_nha.csv (dien_tich_m2 -> gia_trieu_vnd)

Ý tưởng của slide:
  - Huấn luyện mô hình bằng một thuật toán LẶP (vòng lặp / epoch)
  - Theo dõi Train error (giảm dần liên tục) và Validation error
    (giảm rồi sau đó TĂNG trở lại do overfitting)
  - "Early Stopping": dừng thuật toán ngay tại vòng lặp mà Validation error
    đạt giá trị NHỎ NHẤT, thay vì train tới khi hội tụ hẳn (dễ overfitting)

Ở đây, mình dùng đa thức bậc CAO (degree=15) - một mô hình quá phức tạp -
và huấn luyện bằng GRADIENT DESCENT (không dùng nghiệm đóng, không regularization)
để tạo ra đúng hiện tượng: vòng lặp càng nhiều -> mô hình càng khớp sát training
set -> càng dễ overfitting nếu train quá lâu.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# BƯỚC 1: ĐỌC DỮ LIỆU, CHIA TRAIN / VALIDATION / TEST
# ============================================================
df = pd.read_csv("du_lieu_gia_nha.csv")

X_raw_all = df[["dien_tich_m2"]].values
y_raw_all = df["gia_trieu_vnd"].values.reshape(-1, 1)

# Sắp xếp theo diện tích tăng dần rồi chia XEN KẼ (mỗi 5 điểm lấy 1 làm test,
# 1 làm validation) để test/validation luôn nằm TRONG phạm vi của train,
# tránh hiện tượng ngoại suy (extrapolation) gây nhiễu khi dữ liệu quá ít (14 điểm)
order = np.argsort(X_raw_all[:, 0])
X_sorted = X_raw_all[order]
y_sorted = y_raw_all[order]

idx_test = np.arange(0, len(X_sorted), 5)          # 0, 5, 10, ...
idx_val = np.arange(2, len(X_sorted), 5)           # 2, 7, 12, ...
idx_train = np.array([i for i in range(len(X_sorted)) if i not in idx_test and i not in idx_val])

X_train_raw, y_train = X_sorted[idx_train], y_sorted[idx_train]
X_val_raw, y_val = X_sorted[idx_val], y_sorted[idx_val]
X_test_raw, y_test = X_sorted[idx_test], y_sorted[idx_test]

print(f"Train set (Z)     : {len(X_train_raw)} điểm")
print(f"Validation set (Y): {len(X_val_raw)} điểm")
print(f"Test set          : {len(X_test_raw)} điểm")

# ============================================================
# BƯỚC 2: TẠO ĐẶC TRƯNG ĐA THỨC BẬC CAO (degree=15) + CHUẨN HOÁ
#          (chuẩn hoá dựa trên TRAIN set, áp dụng lại cho val/test)
# ============================================================
DEGREE = 6
poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
X_train_poly = poly.fit_transform(X_train_raw)
X_val_poly = poly.transform(X_val_raw)
X_test_poly = poly.transform(X_test_raw)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_poly)
X_val_s = scaler.transform(X_val_poly)
X_test_s = scaler.transform(X_test_poly)

# Chuẩn hoá luôn y để gradient descent ổn định hơn (không đổi bản chất bài toán)
y_scaler = StandardScaler()
y_train_s = y_scaler.fit_transform(y_train)
y_val_s = y_scaler.transform(y_val)
y_test_s = y_scaler.transform(y_test)

n_train, n_features = X_train_s.shape

# ============================================================
# BƯỚC 3: HUẤN LUYỆN BẰNG GRADIENT DESCENT, GHI LẠI ERROR MỖI VÒNG LẶP
#          KHÔNG dùng regularization -> mô hình sẽ overfitting dần theo thời gian
# ============================================================
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def train_adam(n_iters, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, record_every=5):
    """Huấn luyện bằng Adam optimizer (một biến thể mượt của Gradient Descent),
    không có regularization -> mô hình bậc cao sẽ overfitting dần theo vòng lặp."""
    w = np.zeros((n_features, 1))
    b = 0.0
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    mb, vb = 0.0, 0.0

    history = {"iter": [], "w": [], "b": [], "train": [], "val": [], "test": []}

    for it in range(1, n_iters + 1):
        error = (X_train_s @ w + b) - y_train_s
        grad_w = (2.0 / n_train) * (X_train_s.T @ error)
        grad_b = (2.0 / n_train) * np.sum(error)

        # Gradient clipping: giữ ổn định số học khi bậc đa thức rất cao
        grad_norm = np.sqrt(np.sum(grad_w ** 2) + grad_b ** 2)
        max_norm = 5.0
        if grad_norm > max_norm:
            scale = max_norm / (grad_norm + 1e-12)
            grad_w = grad_w * scale
            grad_b = grad_b * scale

        mw[:] = beta1 * mw + (1 - beta1) * grad_w
        vw[:] = beta2 * vw + (1 - beta2) * (grad_w ** 2)
        mw_hat = mw / (1 - beta1 ** it)
        vw_hat = vw / (1 - beta2 ** it)
        w -= lr * mw_hat / (np.sqrt(vw_hat) + eps)

        mb = beta1 * mb + (1 - beta1) * grad_b
        vb = beta2 * vb + (1 - beta2) * (grad_b ** 2)
        mb_hat = mb / (1 - beta1 ** it)
        vb_hat = vb / (1 - beta2 ** it)
        b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        if it % record_every == 0:
            history["iter"].append(it)
            history["w"].append(w.copy())
            history["b"].append(b)
            history["train"].append(mse(y_train_s, X_train_s @ w + b))
            history["val"].append(mse(y_val_s, X_val_s @ w + b))
            history["test"].append(mse(y_test_s, X_test_s @ w + b))

    return history


N_ITERS = 8000
RECORD_EVERY = 5
history = train_adam(N_ITERS, lr=0.006, record_every=RECORD_EVERY)

iters_recorded = np.array(history["iter"])
train_errors = np.array(history["train"])
val_errors = np.array(history["val"])
test_errors = np.array(history["test"])

train_errors = np.array(train_errors)
val_errors = np.array(val_errors)
test_errors = np.array(test_errors)
iters_recorded = np.array(iters_recorded)

# ============================================================
# BƯỚC 4: EARLY STOPPING - tìm vòng lặp mà Validation error NHỎ NHẤT
# ============================================================
best_idx = int(np.argmin(val_errors))
best_iter = iters_recorded[best_idx]

print(f"\n===== EARLY STOPPING =====")
print(f"Tổng số vòng lặp đã chạy : {N_ITERS}")
print(f"Điểm dừng sớm (early stop) tại vòng lặp: {best_iter}")
print(f"  Train error tại đó      : {train_errors[best_idx]:.5f}")
print(f"  Validation error tại đó : {val_errors[best_idx]:.5f} (NHỎ NHẤT)")
print(f"  Test error tại đó       : {test_errors[best_idx]:.5f}")

print(f"\nSo sánh nếu KHÔNG dừng sớm (train hết {N_ITERS} vòng lặp):")
print(f"  Train error cuối cùng      : {train_errors[-1]:.5f}")
print(f"  Validation error cuối cùng : {val_errors[-1]:.5f}  <-- TĂNG so với lúc dừng sớm => OVERFITTING")
print(f"  Test error cuối cùng       : {test_errors[-1]:.5f}  <-- TĂNG so với lúc dừng sớm => OVERFITTING")

# Lấy lại trọng số w, b đã lưu sẵn trong lịch sử huấn luyện
# (không cần train lại) - tại điểm early stopping và tại vòng lặp cuối cùng
w_es, b_es = history["w"][best_idx], history["b"][best_idx]
w_final, b_final = history["w"][-1], history["b"][-1]  # bị overfitting (train hết N_ITERS vòng)


def predict_original_scale(x_raw_grid, w_use, b_use):
    Xp = poly.transform(x_raw_grid)
    Xs = scaler.transform(Xp)
    y_s = Xs @ w_use + b_use
    return y_scaler.inverse_transform(y_s)


# ============================================================
# VẼ ĐỒ THỊ
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(11, 12), gridspec_kw={"height_ratios": [1.3, 1]})

# --- (1) ĐỒ THỊ CHÍNH: 3 đường Train/Validation/Test error theo vòng lặp ---
ax = axes[0]
ax.plot(iters_recorded, train_errors, color="blue", linewidth=2, label="Train error")
ax.plot(iters_recorded, val_errors, color="red", linewidth=2, label="Validation error")
ax.plot(iters_recorded, test_errors, color="green", linewidth=2, label="Test error")

ax.axvline(best_iter, color="black", linestyle="--", linewidth=1.3)
ax.scatter([best_iter], [val_errors[best_idx]], color="black", zorder=5, s=60)
ax.annotate(
    f"EARLY STOPPING\nvòng lặp = {best_iter}\nval error nhỏ nhất",
    xy=(best_iter, val_errors[best_idx]),
    xytext=(best_iter + N_ITERS * 0.12, val_errors[best_idx] * 3.5 + 0.05),
    fontsize=9,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="black"),
)

ax.axvspan(0, best_iter, color="green", alpha=0.05)
ax.axvspan(best_iter, N_ITERS, color="red", alpha=0.05)
ax.text(best_iter * 0.4, ax.get_ylim()[1] * 0.9, "Underfitting -> Vừa fit", fontsize=9, color="darkgreen")
ax.text(best_iter + (N_ITERS - best_iter) * 0.3, ax.get_ylim()[1] * 0.9, "OVERFITTING", fontsize=9, color="darkred", fontweight="bold")

ax.set_xlabel("Số vòng lặp (iteration) của Gradient Descent")
ax.set_ylabel("Error (MSE, trên dữ liệu đã chuẩn hoá)")
ax.set_title(
    f"EARLY STOPPING: 3 đường Train/Validation/Test error xuyên suốt quá trình huấn luyện\n"
    f"(degree={DEGREE}, không regularization, học bằng Gradient Descent)"
)
ax.legend(loc="upper right")
ax.set_yscale("log")

# --- (2) So sánh đường mô hình: TRƯỚC (overfit) và SAU (early stopping) ---
ax2 = axes[1]
x_plot = np.linspace(X_raw_all.min() - 5, X_raw_all.max() + 5, 300).reshape(-1, 1)

ax2.scatter(X_train_raw, y_train, color="red", label="Training samples", zorder=3)
ax2.scatter(X_val_raw, y_val, color="orange", label="Validation samples", zorder=3)
ax2.scatter(X_test_raw, y_test, color="green", marker="s", label="Test samples", zorder=3)

y_overfit_plot = predict_original_scale(x_plot, w_final, b_final)
y_es_plot = predict_original_scale(x_plot, w_es, b_es)

ax2.plot(x_plot, y_overfit_plot, color="crimson", linestyle="--", linewidth=1.8,
         label=f"Mô hình OVERFIT (sau {N_ITERS} vòng lặp)")
ax2.plot(x_plot, y_es_plot, color="blue", linewidth=2.2,
         label=f"Mô hình ĐÃ SỬA (Early Stopping tại vòng {best_iter})")

ax2.set_xlabel("Diện tích (m2)")
ax2.set_ylabel("Giá (triệu VND)")
ax2.set_title("So sánh đường dự đoán: trước và sau khi áp dụng Early Stopping")
ax2.legend(fontsize=8, loc="upper left")
ax2.set_ylim(y_raw_all.min() - 1500, y_raw_all.max() + 1500)

plt.tight_layout()
out_path = "early_stopping_demo.png"
plt.savefig(out_path, dpi=130)
print(f"\nĐã lưu biểu đồ tại: {out_path}")