import time
import os
import numpy as np  # 💡 確保匯入 numpy
from dotenv import load_dotenv

# 匯入自定義模組
from src.inference import PPEEdgeInference
from src.utils import PerformanceTimer, MQTTClient

# 載入環境變數
load_dotenv('config/.env')
MODEL_PATH = os.getenv("MODEL_PATH", "models/ppe_model_quantized.tflite")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))

def start_project():
    print("=== 02.ai_rpi_ppe_edge_TF_Lite 系統啟動 (邊緣優化版) ===", flush=True)
    print(f"📡 診斷資訊: 模型路徑={MODEL_PATH}, 相機索引={CAMERA_INDEX}", flush=True)

    engine = None
    last_alert_time = 0
    ALERT_COOLDOWN = 2

    try:
        # 1. 初始化組件
        engine = PPEEdgeInference(MODEL_PATH, CAMERA_INDEX)
        timer = PerformanceTimer()
        mqtt_bus = MQTTClient()
        mqtt_bus.connect()
        print("✅ 所有系統組件初始化完成", flush=True)

        while True:
            # 影像採集
            frame = engine.capture_with_healing()
            
            # 💡 核心修正 A：顯式判斷，絕對不寫 if frame:
            if frame is None:
                time.sleep(0.1)
                continue

            # 2. AI 推論階段
            # 💡 如果 preprocess 裡面也有 if frame: 就會在那裡崩潰
            input_tensor = engine.preprocess(frame)
            raw_results = engine.infer(input_tensor)
            detections = engine.postprocess(raw_results, THRESHOLD)

            # 3. 邏輯判定與警報 (確保 detections 是一個 list)
            if isinstance(detections, list):
                for det in detections:
                    # 偵測到違規 (class_id 1: 無安全帽)
                    if det.get('class_id') == 1: 
                        current_time = time.time()
                        if current_time - last_alert_time > ALERT_COOLDOWN:
                            mqtt_bus.publish_alert({
                                "event": "No_Helmet", 
                                "score": round(float(det['score']), 2),
                                "timestamp": current_time
                            })
                            print(f"🚨 警報發送：偵測到未戴安全帽！信心度: {det['score']:.2f}", flush=True)
                            last_alert_time = current_time

            # 4. 效能更新
            timer.update_frame_count()
            if timer.frame_count % 30 == 0:
                print(f"📊 目前效能狀態: {timer.get_fps()} FPS", flush=True)

    except Exception as e:
        print(f"🔥 系統運行中斷: {e}", flush=True)
    finally:
        if engine is not None:
            engine.release()
        print("🔌 資源已釋放，系統終止", flush=True)

if __name__ == "__main__":
    start_project()