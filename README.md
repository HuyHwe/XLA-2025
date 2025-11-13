# MNIST Handwriting Recognition Web App

Ứng dụng web nhận diện chữ viết tay dựa trên mô hình CNN tự xây dựng bằng NumPy. Hệ thống gồm:

- **Notebook huấn luyện** (`XLA.ipynb`): định nghĩa các lớp CNN thủ công, huấn luyện trên MNIST, lưu mô hình `cnn_model.pkl`.
- **Backend FastAPI** (`backend/`): nạp mô hình, cung cấp API `/predict`.
- **Frontend React** (`frontend/`): giao diện canvas cho người dùng vẽ và gửi ảnh đến API.

## 1. Chuẩn Bị Môi Trường

### Backend
```bash
cd XLA-2025
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Frontend
```bash
cd XLA-2025/frontend
npm install
```

## 2. Chạy Ứng Dụng

### Backend
```bash
cd XLA-2025
uvicorn backend.main:app --host 127.0.0.1 --port 9000 --reload
```

- API hoạt động tại `http://127.0.0.1:9000`.
- Swaggers docs: `http://127.0.0.1:9000/docs`.
- Endpoint chính:
  - `POST /predict`: nhận JSON `{ "image_base64": "<data_url>" }`, trả `{ "prediction": <digit> }`.

### Frontend
```bash
cd XLA-2025/frontend
npm start
```
- Giao diện chạy ở `http://localhost:3000`.
- Vẽ chữ số 0–9, nhấn “Nhận diện” để xem kết quả.

## 3. Kiến Trúc Backend

- `main.py`: FastAPI app, nạp model một lần khi startup.
- `model_defs.py`: tái định nghĩa các lớp Convolution/MaxPool/Fully_Connected cho việc unpickle.
- `_preprocess_image`: chuyển ảnh sang grayscale, resize 28×28, chuẩn hoá `[0,1]`.

## 4. Frontend React

- `src/components/DrawingBoard.jsx`: canvas vẽ bằng chuột, gửi dữ liệu base64 qua fetch.
- `src/styles/`: `global.css` và `AppLayout.css` cấu hình nền, font, bố cục.
- Tạo project bằng CRA, đã loại bỏ file boilerplate.

## 5. Huấn Luyện / Điều Chỉnh Mô Hình

- Notebook `XLA.ipynb` mô tả toàn bộ quy trình huấn luyện và lưu `cnn_model.pkl`.
- Muốn huấn luyện lại: chạy notebook, cập nhật file `.pkl` trong thư mục `XLA-2025`.


## 6. Cấu Trúc Thư Mục

```
XLA-2025/
├─ XLA.ipynb
├─ cnn_model.pkl
├─ README.md
├─ requirements.txt        # Nhiều thư viện tham khảo
├─ backend/
│  ├─ main.py
│  ├─ model_defs.py
│  ├─ requirements.txt
│  └─ __init__.py
└─ frontend/
   ├─ package.json
   ├─ public/
   └─ src/
      ├─ App.jsx
      ├─ index.js
      ├─ components/
      │  └─ DrawingBoard.jsx
      └─ styles/
         ├─ global.css
         └─ AppLayout.css
```



