import time
import os
from dotenv import load_dotenv
from src.inference import PPEEdgeInference # 筆記 P.14
from src.utils import PerformanceTimer, MQTTClient # 筆記 P.14

load_dotenv('config/.env')
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))

def start_project():
    print("=== 02.ai_rpi_ppe_edge_TF_Lite 系統啟動 ===")
    
    # 初始化組件 [cite: 426, 427, 429]
    engine = PPEEdgeInference(CAMERA_INDEX)
    timer = PerformanceTimer()
    mqtt_bus = MQTTClient()
    mqtt_bus.connect()

    try:
        while True:
            # 1. 採集與自癒 [cite: 436, 437]
            frame = engine.capture_with_healing()
            if frame is None:
                time.sleep(1)
                continue

            # 2. AI 推論流程 [cite: 440, 441, 442]
            input_tensor = engine.preprocess(frame)
            raw_results = engine.infer(input_tensor)
            detections = engine.postprocess(raw_results)

            # 3. 結果判定與警報發送 [cite: 444, 447]
            for det in detections:
                if det['class_id'] == 1: # 假設 1 是未戴安全帽
                    mqtt_bus.publish_alert({
                        "event": "No_Helmet", 
                        "score": round(det['score'], 2),
                        "timestamp": time.time()
                    })
                    print(f"🚨 發現違規！信心度: {det['score']:.2f}")

            # 4. 定期輸出 FPS 數據 [cite: 453]
            if timer.frame_count % 30 == 0:
                print(f"📊 目前狀態: {timer.get_fps()}")

    except Exception as e:
        print(f"🔥 系統異常: {e}")
    finally:
        engine.release()
        print("🔌 資源釋放")

if __name__ == "__main__":
    start_project()