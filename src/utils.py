import cv2
import time
import paho.mqtt.client as mqtt
import json
import os

class MQTTClient:
    def __init__(self):
        # 從環境變數讀取配置，確保資安
        self.broker = os.getenv("MQTT_BROKER", "127.0.0.1")
        self.port = int(os.getenv("MQTT_PORT", 1883))
        self.topic = os.getenv("MQTT_TOPIC", "factory/ppe/alerts")
        
        self.client = mqtt.Client()
        # 實作指數退避重連邏輯，確保通訊韌性
        self.client.reconnect_delay_set(min_delay=1, max_delay=120)
        
    def connect(self):
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            print(f"📡 MQTT 已連接至 {self.broker}")
        except Exception as e:
            print(f"❌ MQTT 連接失敗: {e}")

    def publish_alert(self, detection_info):
        """ 發送工安警報 JSON 數據 """
        payload = json.dumps(detection_info)
        self.client.publish(self.topic, payload)
        
class PerformanceTimer:
    """ 計算 FPS 與推論延遲，提供數據給面試官觀看 """
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0

    def get_fps(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        return f"FPS: {fps:.2f}"

def draw_info(frame, info_text, color=(0, 255, 0)):
    """ 在畫面上標註資訊 """
    cv2.putText(frame, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame