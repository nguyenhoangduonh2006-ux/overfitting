"""
Du bao gia nha bang Hoi quy tuyen tinh (Linear Regression)
------------------------------------------------------------
Bai toan: Dua vao dien tich, so phong ngu, khoang cach den trung tam
          -> du doan gia nha (trieu VND)

Cach chay:
    pip install pandas scikit-learn
    python hoi_quy_tuyen_tinh.py
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,


# BUOC 1: Doc du lieu tu file CSV
# ---------------------------------------------------
du_lieu = pd.read_csv("du_lieu_gia_nha.csv")
print("5 dong du lieu dau tien:")
print(du_lieu.head())


# BUOC 2: Tach du lieu thanh dau vao (X) va dau ra (y)
# ---------------------------------------------------
# X: cac yeu to dung de du doan (dien tich, so phong ngu, khoang cach)
# y: gia nha - cai ma chung ta muon du doan
X = du_lieu[["dien_tich_m2", "so_phong_ngu", "khoang_cach_trung_tam_km"]]
y = du_lieu["gia_trieu_vnd"]


# BUOC 3: Chia du lieu thanh tap huan luyen (train) va tap kiem tra (test)
# ---------------------------------------------------
# 80% du lieu de "hoc" mo hinh, 20% de "kiem tra" xem mo hinh du doan tot khong
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# BUOC 4: Tao va huan luyen mo hinh Hoi quy tuyen tinh
# ---------------------------------------------------
mo_hinh = LinearRegression() # tạo mô hình huấn luyện
mo_hinh.fit(X_train, y_train)   # mo hinh "hoc" moi quan he giua X va y


# BUOC 6: Du doan tren tap kiem tra va danh gia do chinh xac
# ---------------------------------------------------
y_du_doan = mo_hinh.predict(X_test)

sai_so_trung_binh = mean_absolute_error(y_test, y_du_doan)


print(f"\nSai so trung binh (MAE): {sai_so_trung_binh:.2f} trieu VND")



# BUOC 7: Thu du doan cho 1 can nha moi
# ---------------------------------------------------
nha_moi = pd.DataFrame({
    "dien_tich_m2": [82],
    "so_phong_ngu": [2],
    "khoang_cach_trung_tam_km": [3.5],
})
gia_du_doan = mo_hinh.predict(nha_moi)[0]
print(f"\nDu doan gia cho can nha 82m2, 2 phong ngu, cach trung tam 3.5km:")
print(f"  -> Khoang {gia_du_doan:.0f} trieu VND")
