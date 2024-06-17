import base64
from PIL import Image, ImageDraw
import numpy as np
import io
import cv2
import requests
from dotenv import load_dotenv
import os
"""
city scapes dataset
    "0": "road",
    "1": "sidewalk",
    "2": "building",
    "3": "wall",
    "4": "fence",
    "5": "pole",
    "6": "traffic light",
    "7": "traffic sign",
    "8": "vegetation",
    "9": "terrain",
    "10": "sky",
    "11": "person",
    "12": "rider",
    "13": "car",
    "14": "truck",
    "15": "bus",
    "16": "train",
    "17": "motorcycle",
    "18": "bicycle"
"""
def calculate_pixel_area(mask_data):
    areas = []
    
    for item in mask_data:
        print("SEG TEST")
        print(item)
        mask_bytes = base64.b64decode(item['mask'])
        mask_image = Image.open(io.BytesIO(mask_bytes))
        mask_array = np.array(mask_image)
        pixel_count = np.count_nonzero(mask_array)

        areas.append({'label': item['label'], 'pixels': pixel_count, 'mask': item['mask']})
    return areas

def overlay_multiple_masks_on_image(mask_data, indices, original_image_path, colors, alpha=0.5):
    # Load the original image
    original_image = Image.open(original_image_path).convert("RGBA")

    # Create a blank RGBA image for the mask
    mask_rgba = Image.new("RGBA", original_image.size)

    # Decode and process each mask
    for index, color in zip(indices, colors):
        item = mask_data[index]

        # Decode the base64 string to bytes for the mask
        mask_bytes = base64.b64decode(item['mask'])
        mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")

        # Convert mask image to numpy array
        mask_array = np.array(mask_image)

        # Overlay each mask with the specified color
        for y in range(mask_image.height):
            for x in range(mask_image.width):
                if mask_array[y, x] > 0:  # If the pixel is part of the mask
                    mask_rgba.putpixel((x, y), color + (int(255 * alpha),))

    # Combine the original image with the mask
    combined = Image.alpha_composite(original_image, mask_rgba)

    # Draw the borders around the mask areas
    draw = ImageDraw.Draw(combined)
    for index, color in zip(indices, colors):
        item = mask_data[index]
        mask_bytes = base64.b64decode(item['mask'])
        mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_array = np.array(mask_image)
        
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2:
                contour = [tuple(pt) for pt in contour]
                draw.line(contour + [contour[0]], fill=color + (255,), width=3)

    return combined


def get_bounding_box_from_mask(base64_str):
    # Decode the base64 string to get image bytes
    image_bytes = base64.b64decode(base64_str)
    
    # Create an image from the bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert the image to a numpy array
    mask = np.array(image)
    
    # Ensure the mask is binary
    mask = mask > 0
    
    # Find the coordinates where the mask is True
    coords = np.column_stack(np.where(mask))
    
    # Get the bounding box coordinates
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return x_min, y_min, x_max, y_max

################################################
# for query
def query(filename):
    "query to huggingface inference api"

    load_dotenv(dotenv_path=".env")
    API_URL = os.getenv('API_URL_SEG')
    HF_token = os.getenv('HF_token')
    headers = {"Authorization": f"Bearer {HF_token}"}

    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)

    return response.json()


def visualize_bbox(input_image, car_bbox):
    output_image = np.array(input_image.copy())

    xmin, ymin, xmax, ymax = car_bbox
    color = (255, 255, 255)
    thickness = 3
    cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, thickness)

    return Image.fromarray(output_image)

################################################
# main process
def process(image_path):
    output = query(image_path)

    if 'error' in output:
        return None, None, None, None, output['error']

    areas = calculate_pixel_area(output)
    car_index = []
    road_index = None
    sidewalk_index = None

    for i in range(len(areas)):
        label = areas[i]['label']
        print(f"{i}. Label: {label}, Pixels: {areas[i]['pixels']}")
        if label in ['car', 'truck', 'bus', 'motorcycle'] :
            car_index.append(i)
        elif label == 'road':
            road_index = i
        elif label == 'sidewalk':
            sidewalk_index = i
    
    # Sort by area
    car_index.sort(key=lambda idx: areas[idx]['pixels'], reverse=True)

    indices = []
    colors = []

    seg_label = []
    if car_index:
        seg_label.append(areas[car_index[0]]['label'])
    if sidewalk_index:
        seg_label.append('sidewalk')
    if road_index:
        seg_label.append('road')
    

    if road_index is not None:
        indices.append(road_index)
        colors.append((255, 0, 0))  # Red for road

    if sidewalk_index is not None:
        indices.append(sidewalk_index)
        colors.append((0, 255, 0))  # Green for sidewalk

    if car_index:
        indices.append(car_index[0])
        colors.append((0, 0, 255))  # Blue for largest car

    if indices:
        output_image = overlay_multiple_masks_on_image(mask_data=output,
                                                       indices=indices,
                                                       original_image_path=image_path,
                                                       colors=colors)
    
    else:
        return None, None, None, seg_label, "can't detect car/road/sidewalk (in SEG)"
    
    if len(car_index) == 0:
        return output_image, areas, None, seg_label, "can't detect car(or vehicles) (in SEG)"

    car_bbox = get_bounding_box_from_mask(areas[car_index[0]]['mask'])
    output_image = visualize_bbox(output_image, car_bbox)

    

    return output_image, areas, car_bbox, seg_label, None