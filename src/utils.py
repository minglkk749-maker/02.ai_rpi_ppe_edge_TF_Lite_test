import paho.mqtt.client as mqtt # 筆記 P.13
import json
import time
import os

class MQTTClient:
    def __init__(self):
        self.broker = os.getenv("MQTT_BROKER", "10.142.83.47")
        self.port = int(os.getenv("MQTT_PORT", 1883))
        self.topic = os.getenv("MQTT_TOPIC", "factory/ppe/alerts")
        self.client = mqtt.Client()
        # 實作指數退避重連，確保通訊韌性 [cite: 405, 406]
        self.client.reconnect_delay_set(min_delay=1, max_delay=120)

    def connect(self):
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            print(f"📡 MQTT 已連接至 {self.broker}")
        except Exception as e:
            print(f"❌ MQTT 連接失敗: {e}")

    def publish_alert(self, detection_info):
        """發送工安警報 JSON 數據 [cite: 414, 415]"""
        payload = json.dumps(detection_info)
        self.client.publish(self.topic, payload)

class PerformanceTimer:
    """計算 FPS [cite: 318, 319]"""
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0

    def get_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed
        return f"FPS: {fps:.2f}"

def draw_info(frame, info_text, color=(0, 255, 0)):
    """在畫面上標註資訊 (Headless 模式下僅保留邏輯) [cite: 329]"""
    return frame