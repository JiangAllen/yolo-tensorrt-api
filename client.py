import requests
import os
import json

SERVER_URL = "http://localhost:8000/predictimage"
IMAGE_FOLDER = "./src/"

def send_image(image_path):
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(SERVER_URL, files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"🔍 Result ({os.path.basename(image_path)}):")
        print(json.dumps(result, indent=4, ensure_ascii=False))
    else:
        print(f"無法處理 {os.path.basename(image_path)}，錯誤: {response.status_code}, Message: {response.text}")

def process_images_in_folder(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print("image not found")
        return
    print(f"📂 Find {len(image_files)} image(s), Start...")

    for image_name in image_files:
        image_path = os.path.join(folder_path, image_name)
        send_image(image_path)

if __name__ == "__main__":
    process_images_in_folder(IMAGE_FOLDER)
