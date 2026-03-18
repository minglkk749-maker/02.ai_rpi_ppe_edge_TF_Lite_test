import cv2
import numpy as np
import time
import os
from dotenv import load_dotenv

load_dotenv('config/.env')
MODEL_PATH = os.getenv("MODEL_PATH", "models/ppe_model_quantized.tflite")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))

class PPEEdgeInference:
    def __init__(self):
        self.cap = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        self._setup_hardware()
        self._load_edge_model()

    def _setup_hardware(self):
        print(f"🔍 正在初始化相機: {CAMERA_INDEX}...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        # 設定解析度，減輕 RPi 5 負擔
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _load_edge_model(self):
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"🚀 量化模型就緒，輸入尺寸: {self.input_details[0]['shape']}")
        except Exception as e:
            print(f"❌ 模型加載失敗: {e}")

    def preprocess(self, frame):
        """
        影像預處理：將畫面轉換為 AI 專用張量
        類比電梯訊號的濾波與轉換 
        """
        # 1. 取得模型要求的尺寸 (例如 320x320)
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # 2. 縮放影像並轉換顏色空間 (BGR 轉 RGB)
        img = cv2.resize(frame, (width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. 增加維度以符合模型輸入 (H,W,C -> 1,H,W,C)
        input_data = np.expand_dims(img, axis=0)
        
        # 4. 如果模型是浮點數，需要標準化 (Normalize)
        if self.input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5
            
        return input_data

    def postprocess(self, output_data, threshold=0.5):
        """
        將模型輸出的原始數據解析為具體的偵測結果
        threshold: 信心度門檻，過低則視為雜訊
        """
        # 假設模型輸出格式為 [1, 偵測數量, 數據] (依不同模型架構調整)
        # 典型的 TFLite 偵測模型輸出：boxes, classes, scores, count
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        detections = []
        for i in range(len(scores)):
            if scores[i] > threshold:
                detections.append({
                    "box": boxes[i],       # [ymin, xmin, ymax, xmax]
                    "class_id": int(classes[i]),
                    "score": float(scores[i])
                })
        return detections

    def infer(self, input_data):
        """ 執行邊緣端推論 """
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # 取得輸出結果
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def run_loop(self):
        """ 主循環：採集 -> 預處理 -> 推論 -> 顯示 """
        print("▶️ 開始監控流程...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # 實作自癒機制 
                self._setup_hardware()
                continue
                
            # --- 核心 AI 流程 ---
            input_tensor = self.preprocess(frame)
            results = self.infer(input_tensor)
            
            # --- 顯示結果 (暫時僅顯示畫面) ---
            cv2.imshow('PPE Edge Monitor', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = PPEEdgeInference()
    engine.run_loop()