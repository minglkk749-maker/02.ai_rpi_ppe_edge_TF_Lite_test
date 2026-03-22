import cv2
import numpy as np
import os
import tflite_runtime.interpreter as tflite # 筆記 P.9
from dotenv import load_dotenv

# 加載環境變數
load_dotenv('config/.env')
MODEL_PATH = os.getenv("MODEL_PATH", "models/ppe_model_quantized.tflite")

class PPEEdgeInference:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._setup_hardware() # 筆記 P.9
        self._load_edge_model() # 筆記 P.9

    def _setup_hardware(self):
        """初始化相機硬體，具備自癒重連邏輯 [cite: 248, 249]"""
        print(f"正在初始化相機: {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        # 設定解析度，減輕 RPi 5 負擔 [cite: 251]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _load_edge_model(self):
        """載入 TFLite 量化模型 [cite: 252, 255]"""
        try:
            self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"✅ 模型就緒，輸入尺寸: {self.input_details[0]['shape']}")
        except Exception as e:
            print(f"❌ 模型加載失敗: {e}")

    def capture_with_healing(self):
        """影像自癒機制：若採集失敗則重啟硬體 [cite: 291, 292]"""
        ret, frame = self.cap.read()
        if not ret:
            print("⚠️ 相機訊號中斷，嘗試自動復歸...")
            self.cap.release()
            self._setup_hardware()
            return None
        return frame

    def preprocess(self, frame):
        """影像預處理 (筆記 P.9) [cite: 260]"""
        input_shape = self.input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]
        img = cv2.resize(frame, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, axis=0)
        
        if self.input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        return input_data

    def infer(self, input_data):
        """執行邊緣端推論 [cite: 276, 280]"""
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

    def postprocess(self, output_data, threshold=0.5):
        """解析偵測結果 [cite: 367, 369]"""
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        detections = []
        for i in range(len(scores)):
            if scores[i] > threshold:
                detections.append({
                    "class_id": int(classes[i]),
                    "score": float(scores[i]),
                    "box": boxes[i].tolist()
                })
        return detections

    def release(self):
        if self.cap: self.cap.release()