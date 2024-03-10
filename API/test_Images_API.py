import requests
import os

url = ""


image_path = os.path.join("images", "67.jpg")


if not os.path.isfile(image_path):
    print(f"File not found: {image_path}")
else:
    with open(image_path, "rb") as image_file:
        files = {"file": (os.path.basename(image_path), image_file)}
        
        response = requests.post(url, files=files)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
