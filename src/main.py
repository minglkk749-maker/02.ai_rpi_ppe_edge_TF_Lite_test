import cv2
from src.inference import PPEEdgeInference
from src.utils import PerformanceTimer, draw_info, MQTTClient

def start_project():
    print("=== 02.ai_rpi_ppe_edge_TF_Lite 系統啟動 ===")
    
    # 初始化組件
    engine = PPEEdgeInference()
    timer = PerformanceTimer()
    mqtt_bus = MQTTClient()
    mqtt_bus.connect()
    
    try:
        while True:
            frame = engine.capture_with_healing()
            if frame is None: continue # 自癒機制等待中
            
            # 1. AI 推論
            input_tensor = engine.preprocess(frame)
            raw_results = engine.infer(input_tensor)
            detections = engine.postprocess(raw_results)
            
            # 2. 結果處理與警報
            for det in detections:
                # 假設 class_id 1 是「未戴安全帽」
                if det['class_id'] == 1:
                    mqtt_bus.publish_alert({"event": "No_Helmet", "score": det['score']})
                    # 可以在畫面上標註紅色警告 
                    frame = draw_info(frame, "⚠️ ALERT: NO HELMET", color=(0, 0, 255))
            
            # 3. 顯示效能數據 (展示你的邊緣運算價值)
            fps_text = timer.get_fps()
            frame = draw_info(frame, fps_text)
            
            cv2.imshow('Industrial PPE Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    except Exception as e:
        print(f"💥 系統中斷: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_project()