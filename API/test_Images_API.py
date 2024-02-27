import requests
import os

# The URL of your FastAPI application's /search/ endpoint
url = "http://178.254.23.172/api/search/"

# The path to the image file you want to upload, relative to this script's directory
image_path = os.path.join("images", "67.jpg")

# Check if the image file exists
if not os.path.isfile(image_path):
    print(f"File not found: {image_path}")
else:
    # Open the image file in binary mode
    with open(image_path, "rb") as image_file:
        # Define the 'files' parameter of the request to include the image file
        files = {"file": (os.path.basename(image_path), image_file)}
        
        # Make a POST request to the FastAPI endpoint with the image file
        response = requests.post(url, files=files)

        # Print the status code and response data
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
