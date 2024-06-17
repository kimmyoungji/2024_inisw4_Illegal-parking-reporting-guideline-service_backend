import requests
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import easyocr
import re
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')


def query(filename):
    "query to huggingface inference api"

    load_dotenv(dotenv_path=".env")
    API_URL = os.getenv('API_URL_LP')
    HF_token = os.getenv('HF_token')
    headers = {"Authorization": f"Bearer {HF_token}"}

    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)

    return response.json()

def process(input_path, car_bbox, seg_image):
    reader = easyocr.Reader(['ko'], gpu=True)
    print("OCR ready")
    
    original_image = Image.open(input_path).convert("RGBA")
    cropped_car = original_image.crop(car_bbox)
    cropped_car.save("/tmp/cropped_car.png")
    
    output = query("/tmp/cropped_car.png")
    
    if 'error' in output:
        return None, None, output['error']

    print(output)
    
    if len(output) == 0:
        return None, None, "can't detect license plate"

    bbox = output[0]['box'].values()
    xmin, ymin, xmax, ymax = map(int, bbox)
    cropped_plate = cropped_car.crop((xmin, ymin, xmax, ymax))
    
    result = reader.readtext(np.asarray(cropped_plate))
    license_number = ""
    for (_, text, _) in result:
        text = re.sub(r'[^가-힣0-9]', '', text)
        license_number += text
    
    """
    draw = ImageDraw.Draw(cropped_car, 'RGBA')
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="green", width=3)
    
    highlighted_image = cropped_car
    """
    car_xmin, car_ymin, car_xmax, car_ymax = car_bbox
    new_xmin = car_xmin + xmin
    new_ymin = car_ymin + ymin
    new_xmax = car_xmin + xmax
    new_ymax = car_ymin + ymax
    
    draw = ImageDraw.Draw(seg_image, 'RGBA')
    draw.rectangle([(new_xmin, new_ymin), (new_xmax, new_ymax)], outline="green", width=2)
    
    seg_lp_image = seg_image

    return seg_lp_image, license_number, None