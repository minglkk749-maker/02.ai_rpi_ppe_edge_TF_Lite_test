# 1. 使用輕量化基礎鏡像，減少 RPi 存儲壓力 
FROM python:3.11-slim-bookworm

# 2. 設定工作目錄 [cite: 137]
WORKDIR /app

# 3. 安裝系統層級依賴 (OpenCV 運作必備) [cite: 138, 140]
# 這裡展現了對硬體相容性的預判 [cite: 139]
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 複製套件清單並安裝 [cite: 144, 145]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt [cite: 146]

# 5. 複製專案內容 [cite: 147, 148]
COPY . .

# 6. 設定環境變數輸出，方便 Log 調查 [cite: 149, 150]
ENV PYTHONUNBUFFERED=1

# 7. 啟動入口 (指向 src/main.py) [cite: 151, 152]
CMD ["python", "src/main.py"]