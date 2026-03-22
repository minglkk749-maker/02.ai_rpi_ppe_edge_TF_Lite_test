import cv2
import numpy as np
import os
import tflite_runtime.interpreter as tflite

class PPEEdgeInference:
    def __init__(self, model_path, camera_index=0):
        self.model_path = model_path
        self.camera_index = camera_index
        self.cap = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        # --- 核心修正：對調初始化順序 ---
        # 1. 先載入 AI 模型 (大腦啟動)
        self._load_edge_model() 
        # 2. 再初始化相機 (眼睛開啟)
        self._setup_hardware()

    def _setup_hardware(self):
        """初始化相機硬體"""
        print(f"📡 正在初始化相機索引: {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        # 設定解析度，減輕 RPi 5 負擔
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✅ 相機硬體初始化完成")

    def _load_edge_model(self):
        """載入 TFLite 量化模型"""
        try:
            print(f"⏳ 正在從 {self.model_path} 分配模型張量 (這可能需要時間)...")
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"✅ 模型就緒，輸入尺寸: {self.input_details[0]['shape']}")
        except Exception as e:
            print(f"❌ 模型加載失敗: {e}")

    def capture_with_healing(self):
        """影像自癒機制"""
        ret, frame = self.cap.read()
        if not ret:
            print("⚠️ 相機訊號中斷，嘗試自動復歸...")
            self.cap.release()
            self._setup_hardware()
            return None
        return frame

    def preprocess(self, frame):
        """影像預處理"""
        input_shape = self.input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]
        img = cv2.resize(frame, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, axis=0)
        
        # 檢查模型是否需要浮點數標準化
        if self.input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5 
        return input_data

    def infer(self, input_data):
        """執行推論"""
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data) 
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index']) 

    def postprocess(self, output_data, threshold=0.5):
        """解析偵測結果"""
        # 獲取偵測座標、類別與信心度
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
        """釋放相機資源"""
        if self.cap:
            self.cap.release()