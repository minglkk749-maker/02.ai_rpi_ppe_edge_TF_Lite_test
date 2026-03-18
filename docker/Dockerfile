# 1. 使用輕量化基礎鏡像，減少邊緣端存儲壓力 
FROM python:3.11-slim-bookworm

# 2. 設定工作目錄
WORKDIR /app

# 3. 安裝系統層級依賴 (OpenCV 必備)
# 這展現了 FAE 對硬體相容性的預判 
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 複製套件清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 複製專案內容
COPY . .

# 6. 設定環境變數輸出，方便 Log 調查 
ENV PYTHONUNBUFFERED=1

# 7. 啟動入口
CMD ["python", "src/main.py"]