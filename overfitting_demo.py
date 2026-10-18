# -*- coding: utf-8 -*-
"""
Bài toán: Dự báo giá nhà - Thể hiện Overfitting và Fix bằng Ridge + StandardScaler 
(Chỉ tính và hiển thị Train Error và Validation Error)
"""

# 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error  # Chỉ cần dùng hàm tính sai số (Error)

# =========================================================================
# BƯỚC 1: ĐỌC DỮ LIỆU TỪ FILE CSV
# =========================================================================
df = pd.read_csv("du_lieu_gia_nha.csv")
X = df[["dien_tich_m2", "so_phong_ngu", "khoang_cach_trung_tam_km"]]
y = df["gia_trieu_vnd"]

# =========================================================================
# BƯỚC 2: CHIA TẬP DỮ LIỆU (TRAIN, VALIDATION, TEST)
# =========================================================================
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# =========================================================================
# BƯỚC 3: TẠO ĐẶC TRƯNG ĐA THỨC BẬC CAO (NGUYÊN NHÂN GÂY OVERFITTING)
# =========================================================================
poly = PolynomialFeatures(degree=6)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

# =========================================================================
# PHẦN 1: HIỆN TƯỢNG OVERFITTING (Dùng Linear Regression thuần túy)
# =========================================================================
model_overfit = LinearRegression()
model_overfit.fit(X_train_poly, y_train) # huan luyen mo hinh

y_pred_train_ov = model_overfit.predict(X_train_poly)
y_pred_val_ov = model_overfit.predict(X_val_poly)

# Tính Train Error và Validation Error (bằng Mean Squared Error)
train_error_ov = mean_squared_error(y_train, y_pred_train_ov)
val_error_ov = mean_squared_error(y_val, y_pred_val_ov)

print("===== 1. KẾT QUẢ KHI BỊ OVERFITTING (Học vẹt) =====")
print(f"Train Error: {train_error_ov:.2f}")
print(f"Val Error:   {val_error_ov:.2f}")
print("-> Nhận xét: Train Error cực kỳ nhỏ, nhưng Val Error vọt lên khổng lồ do overfit!\n")

# =========================================================================
# PHẦN 2: SAU KHI FIX (Dùng StandardScaler + Ridge Regularization)
# =========================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_val_scaled = scaler.transform(X_val_poly)

model_fixed = Ridge(alpha=50.0)
model_fixed.fit(X_train_scaled, y_train)

y_pred_train_fx = model_fixed.predict(X_train_scaled)
y_pred_val_fx = model_fixed.predict(X_val_scaled)

# Tính lại Error sau khi đã fix
train_error_fx = mean_squared_error(y_train, y_pred_train_fx)
val_error_fx = mean_squared_error(y_val, y_pred_val_fx)

print("===== 2. KẾT QUẢ SAU KHI FIX BẰNG RIDGE + SCALER =====")
print(f"Train Error: {train_error_fx:.2f}")
print(f"Val Error:   {val_error_fx:.2f}")
