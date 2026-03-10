import requests
import base64
import os

def test_health():
    print("Testing /health...")
    response = requests.get("http://localhost:8000/")
    print(f"Status: {response.status_code}, Body: {response.json()}")
    assert response.status_code == 200

def test_detect():
    print("\nTesting /detect...")
    # Use a real image from the test set
    image_path = r"c:\Users\user\Desktop\spinach.v11i.yolov8\test\images\-010_jpeg.rf.87148b8fe260e4bce0e48ad4ab66aa85.jpg"
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {"image": img_base64}
    response = requests.post("http://localhost:8000/detect", json=payload)
    print(f"Status: {response.status_code}, Body: {response.json()}")
    assert response.status_code == 200

def test_recommend_recipes():
    print("\nTesting /api/recommend-recipes...")
    payload = {"ingredients": ["spinach", "egg"]}
    response = requests.post("http://localhost:8000/api/recommend-recipes", json=payload)
    print(f"Status: {response.status_code}, Body: {response.json()}")
    assert response.status_code == 200

def test_generate_recipe():
    print("\nTesting /api/generate-recipe...")
    payload = {"selected_ingredients": ["spinach", "garlic"]}
    response = requests.post("http://localhost:8000/api/generate-recipe", json=payload)
    print(f"Status: {response.status_code}, Body: {response.json()}")
    assert response.status_code == 200

if __name__ == "__main__":
    try:
        test_health()
        test_detect()
        test_recommend_recipes()
        test_generate_recipe()
        print("\nAll system tests passed! ✅")
    except Exception as e:
        print(f"\nTests failed: {e} ❌")
