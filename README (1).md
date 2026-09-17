# Demo: Overfitting, Validation & Early Stopping trên bài toán dự đoán giá nhà

Project nhỏ minh hoạ trực quan các khái niệm **Overfitting**, **Validation**,
**Regularization** và **Early Stopping** trong Machine Learning, dựa theo nội
dung slide *"Chapter 2.2 — Overfitting"*. Bài toán ví dụ: dự đoán **giá nhà**
dựa trên **diện tích** bằng **Polynomial Regression**.

## Mục lục

- [Ý tưởng chung](#ý-tưởng-chung)
- [Cấu trúc project](#cấu-trúc-project)
- [Cài đặt](#cài-đặt)
- [1. `overfitting_demo.py` — Overfitting & Ridge Regularization](#1-overfitting_demopy--overfitting--ridge-regularization)
- [2. `early_stopping_demo.py` — Overfitting & Early Stopping](#2-early_stopping_demopy--overfitting--early-stopping)
- [Giải thích khái niệm](#giải-thích-khái-niệm)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

## Ý tưởng chung

Cả 2 script đều làm theo đúng quy trình trong slide:

1. Chia dữ liệu thành **Training set / Validation set / Test set**.
2. Cố tình xây dựng một mô hình **quá phức tạp** (đa thức bậc cao) để tạo ra
   hiện tượng **overfitting**: train error rất thấp nhưng validation/test
   error rất cao.
3. Áp dụng **một kỹ thuật chống overfitting** lấy từ slide để khắc phục, rồi
   so sánh kết quả trước/sau bằng số liệu và biểu đồ.

| Script | Kỹ thuật chống overfitting minh hoạ |
|---|---|
| `overfitting_demo.py` | Chọn bậc đa thức qua **Validation** + **Ridge Regularization** (L2) |
| `early_stopping_demo.py` | **Early Stopping** (dừng huấn luyện đúng lúc) |

## Cấu trúc project

```
.
├── du_lieu_gia_nha.csv         # Dữ liệu: diện tích, số phòng ngủ, khoảng cách trung tâm, giá nhà
├── overfitting_demo.py         # Demo: Overfitting -> chọn bậc đa thức qua Validation -> Ridge Regularization
├── overfitting_demo.png        # Biểu đồ kết quả của overfitting_demo.py
├── early_stopping_demo.py      # Demo: Overfitting -> Early Stopping
├── early_stopping_demo.png     # Biểu đồ kết quả của early_stopping_demo.py
└── README.md
```

### Dữ liệu (`du_lieu_gia_nha.csv`)

| Cột | Ý nghĩa |
|---|---|
| `dien_tich_m2` | Diện tích nhà (m²) |
| `so_phong_ngu` | Số phòng ngủ |
| `khoang_cach_trung_tam_km` | Khoảng cách tới trung tâm thành phố (km) |
| `gia_trieu_vnd` | Giá nhà (triệu VNĐ) — biến mục tiêu |

> Cả 2 script chỉ dùng cột `dien_tich_m2` làm đặc trưng đầu vào (mô hình 1
> biến) để có thể **vẽ trực tiếp đường cong đa thức** lên đồ thị 2D, giống hệt
> ví dụ trực quan trong slide gốc.

## Cài đặt

Yêu cầu Python 3.9+ và các thư viện sau:

```bash
pip install numpy pandas matplotlib scikit-learn
```

> Trên Windows, nếu dùng `pip` không cài được (thường gặp với bản Python quá
> mới như 3.14), khuyến nghị dùng Python 3.11 hoặc 3.12.

Chạy từng script (đảm bảo `du_lieu_gia_nha.csv` nằm **cùng thư mục**):

```bash
python overfitting_demo.py
python early_stopping_demo.py
```

Mỗi script sẽ in kết quả số liệu ra terminal và lưu 1 file `.png` biểu đồ
cùng thư mục.

## 1. `overfitting_demo.py` — Overfitting & Ridge Regularization

**Quy trình:**

1. Chia dữ liệu: Test set (20%) tách riêng từ đầu → phần còn lại chia tiếp
   thành Validation set và Training set mới (theo đúng ký hiệu slide:
   `Z = X \ Y`).
2. **Gây overfitting**: fit `Polynomial Regression` bậc **15**, không
   regularization → train error cực thấp, nhưng validation/test error cực
   cao (đường dự đoán dao động dữ dội ở 2 đầu khoảng dữ liệu).
3. **Cách 1 — chọn độ phức tạp mô hình qua Validation**: thử các bậc đa thức
   từ 1 đến 15, chọn bậc có validation error nhỏ nhất.
4. **Cách 2 — Ridge Regularization (L2)**: giữ nguyên bậc 15 (mô hình phức
   tạp) nhưng thêm số hạng phạt vào hàm mất mát:

   ```
   f_reg(w) = f(w) + λ‖w‖²₂
   ```

   Dò giá trị `λ` tốt nhất bằng validation set → mô hình vẫn bậc cao nhưng
   không còn dao động cực đoan.

**Kết quả (`overfitting_demo.png`)** gồm 4 biểu đồ con:

- (A) Mô hình bị overfitting (bậc 15, không regularization)
- (B) Train/Validation/Test error theo bậc đa thức → xác định điểm
  underfitting/overfitting
- (C) Mô hình sau khi thêm Ridge Regularization (bậc 15, λ tối ưu)
- (D) Validation error theo giá trị λ → xác định λ tối ưu

## 2. `early_stopping_demo.py` — Overfitting & Early Stopping

Vì Early Stopping là kỹ thuật áp dụng **trong lúc huấn luyện lặp** (không
dùng nghiệm đóng như `LinearRegression.fit()`), script này tự cài đặt
**Gradient Descent (Adam optimizer)** để huấn luyện đa thức bậc cao, ghi lại
train/validation/test error ở **mỗi vòng lặp**.

**Quy trình:**

1. Chia dữ liệu train/validation/test (chia **xen kẽ theo diện tích tăng
   dần** thay vì ngẫu nhiên, để tránh test/validation rơi ra ngoài phạm vi
   train khi dữ liệu ít — tránh hiện tượng ngoại suy gây nhiễu kết quả).
2. Chuẩn hoá đặc trưng đa thức (`PolynomialFeatures` + `StandardScaler`).
3. Huấn luyện bằng Adam, **không regularization**, qua nhiều vòng lặp — mô
   hình sẽ overfitting dần nếu train quá lâu.
4. **Early Stopping**: tìm vòng lặp mà validation error đạt giá trị nhỏ
   nhất → dùng trọng số tại đúng vòng lặp đó làm mô hình cuối cùng, bỏ qua
   các vòng lặp sau (dù train error ở đó còn thấp hơn).

**Kết quả (`early_stopping_demo.png`)** gồm 2 biểu đồ con:

- **(1) Train / Validation / Test error theo số vòng lặp**: 3 đường liền
  mạch từ đầu (mô hình còn đơn giản) đến cuối (mô hình bị overfitting nặng),
  có đánh dấu điểm Early Stopping (validation error nhỏ nhất) và tô màu 2
  vùng "chưa overfitting" / "overfitting".
- **(2) So sánh đường dự đoán giá nhà theo diện tích**: đường mô hình train
  hết số vòng lặp (overfitting, đứt nét) so với đường mô hình dừng đúng lúc
  Early Stopping (mượt hơn, tổng quát hơn).

> Lưu ý: 2 đường trong biểu đồ (2) là **2 phiên bản của cùng một mô hình**
> tại 2 thời điểm huấn luyện khác nhau (vòng lặp cuối vs. vòng lặp early
> stopping) — **không phải** train/validation/test error. 3 đường
> train/validation/test error nằm ở biểu đồ (1).

## Giải thích khái niệm

| Khái niệm | Ý nghĩa ngắn gọn |
|---|---|
| **Overfitting** | Mô hình khớp quá sát training set nhưng không tổng quát tốt cho dữ liệu mới (train error thấp, validation/test error cao) |
| **Underfitting** | Mô hình quá đơn giản, không khớp tốt cả training lẫn dữ liệu mới (cả train và validation/test error đều cao) |
| **Validation set** | Tập dữ liệu tách riêng từ training set, dùng để đánh giá và lựa chọn mô hình mà không đụng đến test set |
| **Regularization (Ridge/L2)** | Thêm số hạng phạt `λ‖w‖²` vào hàm mất mát để giữ các hệ số mô hình không quá lớn, giảm overfitting mà không cần giảm độ phức tạp mô hình |
| **Early Stopping** | Dừng thuật toán huấn luyện lặp ngay khi validation error đạt giá trị nhỏ nhất, thay vì train tới khi hội tụ hẳn |

## Tài liệu tham khảo

- Vũ Hữu Tiệp, *Machine Learning cơ bản*, 2018 — https://github.com/tiepvupsu/ebookMLCB
- Blog: https://machinelearningcoban.com
- Slide bài giảng *Chapter 2.2 — Overfitting*, CSE — Thuyloi University
