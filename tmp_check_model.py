
import io
import base64
from ultralytics import YOLO
from PIL import Image

MODEL_PATH = 'c:/Users/user/Desktop/kitchen-ai-main/best.pt'
image_path = r"c:\Users\user\Desktop\spinach.v11i.yolov8\test\images\-010_jpeg.rf.87148b8fe260e4bce0e48ad4ab66aa85.jpg"

try:
    model = YOLO(MODEL_PATH)
    img = Image.open(image_path).convert("RGB")
    results = model(img, verbose=True)
    
    print("\n--- Detection Results ---")
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = float(box.conf[0])
            print(f"Object: {name}, Confidence: {conf:.4f}")
except Exception as e:
    print(f"Error: {e}")
