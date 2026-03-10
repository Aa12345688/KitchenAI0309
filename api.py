import io
import base64
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI
app = FastAPI(title="YOLOv8 Ingredient Detector API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Configuration ===
# Local YOLOv8 model path (.pt file)
MODEL_PATH = 'best.pt'

# Global model variable
model = None

def load_model():
    """Lazy load the model to avoid startup errors if path is missing. Fallback to yolov8n.pt."""
    global model
    try:
        if model is None:
            # Try loading local best.pt first
            if os.path.exists(MODEL_PATH):
                model = YOLO(MODEL_PATH)
                print(f"✅ Success: Model loaded from {MODEL_PATH}")
            else:
                # Fallback to official yolov8n.pt if best.pt is missing
                print(f"⚠️ Warning: {MODEL_PATH} not found. Falling back to default 'yolov8n.pt'...")
                model = YOLO('yolov8n.pt')
                print(f"✅ Success: Official 'yolov8n.pt' loaded.")
    except Exception as e:
        print(f"❌ Critical Error loading model: {e}")
    return model

# === Mock Recipe Database (Taiwan/Asian Cuisine Focus) ===
RECIPE_DB = {
    "tomato": {
        "name": "番茄炒蛋",
        "description": "經典家常菜，營養豐富且製作快速。",
        "steps": ["番茄切塊", "雞蛋打散", "熱鍋炒蛋", "加入番茄翻炒"],
        "matches": ["egg"]
    },
    "spinach": {
        "name": "蒜炒菠菜",
        "description": "簡單清脆的經典綠葉菜做法。",
        "steps": ["菠菜洗淨", "蒜末爆香", "大火快炒"],
        "matches": ["garlic"]
    }
}

class DetectionRequest(BaseModel):
    image: str  # Base64 encoded image string

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": load_model() is not None}

@app.post("/detect")
async def detect(request: DetectionRequest):
    detector = load_model()
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # --- 圖像解碼與處理 (Image Decoding) ---
        encoded = request.image.split(",", 1)[1] if "," in request.image else request.image
        img_data = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        # --- 執行模型辨識 (Run YOLO Inference) ---
        results = detector(img, verbose=False)
        
        # --- 解析偵測結果 (Parse Results) ---
        detections = []
        width, height = img.size
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = detector.names[cls_id] # 取得類別名稱
                conf = float(box.conf[0]) # 取得信心度 (0~1)
                xyxy = box.xyxy[0].tolist() # 取得座標 [x1, y1, x2, y2]
                
                # --- 品質判定邏輯 (Spoiled Logic) ---
                # 如果標籤名稱包含 'rotten'，判定為損壞食材
                is_spoiled = (name.lower() == 'rotten')
                
                # 清洗名稱 (移除 'rotten' 前綴)
                clean_name = name.replace('rotten', '').strip()
                if not clean_name:
                    clean_name = name # Fallback if name was just 'rotten'
                
                detections.append({
                    "class": clean_name,
                    "confidence": conf,
                    "bbox": xyxy,
                    "spoiled": is_spoiled
                })
        
        # 將英文標籤轉換為中文 (簡易對應表)
        label_map = {
            "apple": "蘋果", "banana": "香蕉", "tomato": "番茄",
            "egg": "雞蛋", "carrot": "胡蘿蔔", "broccoli": "青花菜",
            "pork": "豬肉", "beef": "牛肉", "chicken": "雞肉",
            "cabbage": "高麗菜", "potato": "馬鈴薯", "onion": "洋蔥"
        }
        for d in detections:
             d["class"] = label_map.get(d["class"].lower(), d["class"])

        return {
            "detections": detections,
            "width": width,
            "height": height
        }
        
    except Exception as e:
        print(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend-recipes")
async def recommend_recipes(ingredients: dict):
    # If the frontend sends a list, we should probably wrap it or change the signature
    # But for now, let's assume it's a dict or update it to handle the list properly.
    items = ingredients.get("ingredients", [])
    return [
        {
            "id": "rec_001",
            "name": "全能型食材濃湯",
            "match_score": 85,
            "description": "運用現有辨識食材合成的高營養湯品。",
            "image": "https://images.unsplash.com/photo-1547592166-23ac45744acd?q=80&w=800",
            "time": "20 min"
        },
        {
            "id": "rec_002",
            "name": "感測器推薦：清炒食蔬",
            "match_score": 92,
            "description": "保留食材原始資料與口感的最佳烹飪協議。",
            "image": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?q=80&w=800",
            "time": "15 min"
        }
    ]

import google.generativeai as genai
import json

# === 安全驗證與內容過濾 (Safety Logic) ===
INTERNAL_BLACKLIST = ["wd-40", "wd40", "機油", "汽油", "電池", "人肉"] # 絕對禁止
SENSITIVE_KEYWORDS = ["大麻", "hemp", "cannabis", "罌粟", "poppy"] # 敏感但可能食用

def validate_ingredients_basic(ingredients: list):
    """
    初步硬體過濾。
    """
    for item in ingredients:
        clean_item = item.strip().lower()
        if any(bad in clean_item for bad in INTERNAL_BLACKLIST):
            return False
    return True

SAFETY_ERROR_RESPONSE = [
    {
        "id": "safety_error",
        "name": "⛔ 核心安全協議攔截",
        "matchScore": 0,
        "description": "偵測到不可食用或違規物品。為了您的安全，系統拒絕生成方案。",
        "steps": [
            {"title": "安全警告", "description": "請檢查您的選中食材清單，確保皆為可食用食品。"},
            {"title": "操作建議", "description": "請移除不可食用物品（如 WD-40、非食物）後重新操作。"}
        ],
        "image": "https://images.unsplash.com/photo-1594322436404-5a0526db4d13?q=80&w=800",
        "time": "N/A",
        "difficulty": "EASY",
        "category": "mixed",
        "requiredIngredients": []
    }
]

SENSITIVE_WARNING_RESPONSE = [
    {
        "id": "sensitive_warning",
        "name": "⚠️ 敏感食材安全提醒",
        "matchScore": 40,
        "description": "偵測到包含敏感或受限成分的食材 (例如：大麻、罌粟成分)。",
        "steps": [
            {"title": "法律合規性", "description": "請確保該食材在您所在的地區為合法販售且可供食用。"},
            {"title": "食用建議", "description": "若要進行烹飪，請參考專業營養師或合法管道之指南，系統不主動提供細節流程。"},
            {"title": "安全第一", "description": "若不確定來源，請勿輕易嘗試。建議使用常見食材取代。"}
        ],
        "image": "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?q=80&w=800",
        "time": "5 MIN",
        "difficulty": "EASY",
        "category": "mixed",
        "requiredIngredients": []
    }
]

# === 隨機美食圖庫 (Random Food Images) ===
import json
import os

IMAGE_DB_PATH = os.path.join(os.path.dirname(__file__), 'image_db.json')
FOOD_IMAGES = []
try:
    with open(IMAGE_DB_PATH, 'r', encoding='utf-8') as f:
        db_data = json.load(f)
        FOOD_IMAGES = db_data.get('images', [])
except Exception as e:
    print(f"⚠️ 無法載入照片資料庫: {e}")

if not FOOD_IMAGES:
    # 預設保底圖片
    FOOD_IMAGES = ["https://images.unsplash.com/photo-1547592166-23ac45744acd?q=80&w=800"]

# === AI 服務安全介面 (Neural Link Protocol) ===
def call_ai_service(ingredients: list, api_key: str):
    """
    使用 Google GenAI SDK (新版) 生成食譜。
    """
    try:
        # 初始化 SDK (舊版穩定寫法)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        import time
        prompt = f"""
        [角色任務]
        你是一位冷靜、精準且高效的「直覺系主廚」。
        當前執行時間戳記: {time.time()} 
        
        [背景資訊]
        根據食材清單提供實用方案。
        食材清單: {', '.join(ingredients)}
        
        [正常生成指令]
        如果食材皆為正常食物，請嚴格生成 3 道料理且必須包裝在 JSON 數組中，結構如下：
        [
            {{
                "id": "chef_logic_1",
                "name": "[純粹類] 料理名稱",
                "matchScore": 95,
                "description": "食材速報：列出食材與比例。",
                "steps": [
                    {{"title": "準備", "description": "描述步驟"}},
                    {{"title": "執行", "description": "描述步驟"}}
                ],
                "image": "這會由後端自動替換",
                "time": "15 MIN",
                "difficulty": "EASY",
                "category": "vegetable",
                "requiredIngredients": ["食材1"]
            }}
        ]
        
        [約束條件]
        1. 每個料理必須包含 3-5 個 "steps"，每個 step 的 description 不超過 20 字。
        2. 禁止廢話：嚴禁情緒化助詞、寒暄或原理說明。
        3. 字數限制：總回覆 JSON 內容控制在 400 中文字內。
        4. 語言：繁體中文。
        5. 僅輸出 JSON 數據，不帶 Markdown 標記。不要輸出 ```json 標記。
        """
        
        # 發送請求
        response = model.generate_content(prompt)
        
        # 獲取回傳內容
        content = response.text.strip()
        
        # 清洗內容：使用正則表達式尋找 JSON 數組 (Robust extraction)
        import re
        
        # 尋找第一層 JSON Array
        json_match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        else:
            print("⚠️ Regex failed to perfectly match JSON array, attempting raw parse...")
            #有時候外面會包 ```json ... ```，先清掉
            content = content.replace("```json", "").replace("```", "").strip()

        try:
            parsed_recipes = json.loads(content)
            
            # 自動為每一道菜隨機分配一張不同好看的圖片
            import random
            available_images = FOOD_IMAGES.copy()
            for recipe in parsed_recipes:
                if not available_images:
                    available_images = FOOD_IMAGES.copy() # 重置圖片庫
                img_choice = random.choice(available_images)
                recipe["image"] = img_choice
                available_images.remove(img_choice) # 避免重複
                
            return parsed_recipes
            
        except json.JSONDecodeError as decode_e:
            print(f"❌ JSON Parsing Failed: {decode_e}")
            print(f"Raw Content: \n{content}\n")
            return None
        
    except Exception as e:
        import traceback
        print("--- Gemini API Internal Error ---")
        print("--- Gemini API Error Details ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        traceback.print_exc()
        print("--------------------------------")
        return None

@app.post("/api/generate-recipe")
async def generate_recipe(data: dict):
    # 安全載入：從環境變數讀取 OPENAI_API_KEY (實際上是 Gemini Key)
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('VITE_LLM_API_KEY')
    
    # 錯誤處理：若金鑰缺失
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="系統環境配置錯誤 (Neural Link Authentication Failed)"
        )
    
    # 接收前端轉傳的「已勾選食材」
    selected_ingredients = data.get("selected_ingredients", [])
    
    if not selected_ingredients:
        raise HTTPException(status_code=400, detail="未偵測到選中食材")

    # --- Level 1: Python 基礎硬性過濾 (防止極端詞彙) ---
    if not validate_ingredients_basic(selected_ingredients):
        print(f"🚨 Level 1 攔截：{selected_ingredients}")
        return SAFETY_ERROR_RESPONSE

    # --- Level 2: 敏感食材導向 (Sensitive Redirection) ---
    # 如果是能吃但敏感的東西 (例如大麻籽)，給予安全建議而非食譜。
    for item in selected_ingredients:
        if any(skw in item.lower() for skw in SENSITIVE_KEYWORDS):
            print(f"📢 Level 2 敏感引導：{selected_ingredients}")
            return SENSITIVE_WARNING_RESPONSE

    # --- Level 3: 呼叫真實 AI 服務生成食譜 ---
    recipes = call_ai_service(selected_ingredients, api_key)
    
    if recipes:
        print(f"✅ 成功生成 {len(recipes)} 個食譜")
        return recipes
    
    # 如果 AI 失敗，回傳備份數據 (Fallback to mock if AI fails)
    print("⚠️ 呼叫 AI 失敗，使用系統備份協議回傳")
    return [
        {
            "id": f"fallback_{abs(hash(str(selected_ingredients)))}",
            "name": "核心協議：全能型食材濃湯",
            "matchScore": 85,
            "description": "系統目前處於離線快取模式，提供基礎營養方案。",
            "image": "https://images.unsplash.com/photo-1547592166-23ac45744acd?q=80&w=800",
            "time": "25 MIN",
            "difficulty": "MEDIUM",
            "category": "mixed",
            "requiredIngredients": selected_ingredients
        }
    ]

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
