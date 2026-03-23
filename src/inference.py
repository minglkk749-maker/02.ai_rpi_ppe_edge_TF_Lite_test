import cv2
import numpy as np
import os
import time
import tflite_runtime.interpreter as tflite

class PPEEdgeInference:
    def __init__(self, model_path, camera_index=0):
        self.model_path = model_path
        self.camera_index = camera_index
        self.cap = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        # 1. 先啟動大腦
        self._load_edge_model()
        # 2. 再開啟眼睛
        self._setup_hardware()

    def _setup_hardware(self):
        """針對 RPi 5 + Camera Module 3 的初始化"""
        print(f"📡 正在嘗試啟動相機 (索引: {self.camera_index})...", flush=True)
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # 設定解析度
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            print(f"❌ 無法開啟相機設備", flush=True)
            return

        print("⏳ 代理已就緒，等待影像流熱身...", flush=True)
        for i in range(20):
            ret, _ = self.cap.read()
            if ret:
                print(f"✅ 成功擷取第 {i+1} 幀，相機正式就緒！", flush=True)
                return
            time.sleep(0.1)
        print("⚠️ 暖機失敗", flush=True)

    def _load_edge_model(self):
        try:
            print(f"⏳ 正在從 {self.model_path} 分配模型張量...", flush=True)
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"✅ 模型就緒，輸入尺寸: {self.input_details[0]['shape']}", flush=True)
        except Exception as e:
            print(f"❌ 模型加載失敗: {e}", flush=True)

    def capture_with_healing(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def preprocess(self, frame):
        if frame is None:
            return None
            
        input_shape = self.input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]
        
        # 影像縮放與格式轉換
        img = cv2.resize(frame, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 擴充維度 [1, H, W, 3]
        input_data = np.expand_dims(img, axis=0)
        
        # 歸一化處理
        if self.input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5 
        
        return input_data

    def infer(self, input_data):
        if input_data is None:
            return False
        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data) 
            self.interpreter.invoke()
            return True
        except Exception as e:
            print(f"❌ 推論失敗: {e}", flush=True)
            return False

    def postprocess(self, success, threshold=0.5):
        if not success:
            return []

        # 💡 核心修正：使用 .item() 徹底解決 Scalar 轉換報錯
        # 標準模型輸出索引：0:Boxes, 1:Classes, 2:Scores
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] 
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] 
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] 
        
        detections = []
        for i in range(len(scores)): 
            # 確保取得的是純數值
            score_val = scores[i].item() if hasattr(scores[i], 'item') else float(scores[i])
            
            if score_val > threshold: 
                class_id = int(classes[i].item() if hasattr(classes[i], 'item') else classes[i])
                detections.append({
                    "class_id": class_id, 
                    "score": score_val, 
                    "box": boxes[i].tolist() 
                })
        return detections

    def release(self):
        if self.cap:
            self.cap.release()
            print("📷 相機資源已釋放", flush=True)