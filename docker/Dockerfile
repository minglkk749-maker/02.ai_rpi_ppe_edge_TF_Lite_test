FROM python:3.11-slim-bookworm
WORKDIR /app

# 1. 加入 RPi 官方源
RUN apt-get update && apt-get install -y curl gnupg \
    && curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.org/debian/ bookworm main" > /etc/apt/sources.list.d/raspberrypi.list

# 2. 安裝影像庫 (特別加入 libcamera-apps-lite 以防萬一)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libv4l-dev \
    v4l-utils \
    libcamera-v4l2 \
    libcamera-apps-lite \
    && rm -rf /var/lib/apt/lists/*

# 💡 核心修正：直接列出該套件安裝的所有路徑，並列出全系統相關檔案
# 這樣在編譯失敗時，我們看 Build Log 就能看到路徑
RUN echo "--- 檢查套件安裝內容 ---" && \
    dpkg -L libcamera-v4l2 || echo "套件未正確安裝" && \
    echo "--- 全系統搜索 .so 檔案 ---" && \
    find / -name "libcamera-v4l2.so" -o -name "v4l2-compat.so" 2>/dev/null > /app/found_path.txt && \
    cat /app/found_path.txt | grep "." || (echo "❌ 檔案真的不在裡面！" && exit 1)

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "src/main.py"]