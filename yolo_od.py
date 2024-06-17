import os
import torch
from ultralytics import YOLO
from PIL import Image, ImageOps, ImageDraw, ImageFont
import tempfile
from google.cloud import storage



def visualize_detections(image, results, model_names):
    # 이미지의 EXIF 회전 정보 반영
    image = ImageOps.exif_transpose(image)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # 예측 결과 반영
    for box in results[0].boxes:
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()  # xyxy 좌표 형식을 리스트로 변환하여 사용
        conf = box.conf[0].item()  # 신뢰도 값 추출
        cls = int(box.cls[0].item())  # 클래스 번호 추출
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        
        # 사각형 그리기
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
        
        # 텍스트 그리기
        text = f'{model_names[cls]} {conf:.2f}'
        text_size = draw.textsize(text, font=font)
        draw.rectangle([(x_min, y_min - text_size[1]), (x_min + text_size[0], y_min)], fill="red")
        draw.text((x_min, y_min - text_size[1]), text, fill="white", font=font)
    
    return image

def extract_labels(results, model_names):
    detected_objects = []
    for box in results[0].boxes:
        cls = int(box.cls[0].item())  # 클래스 번호 추출
        conf = box.conf[0].item()  # 신뢰도 값 추출
        detected_objects.append((model_names[cls], conf))
    return detected_objects

def download_blob(bucket_name, source_blob_name, destination_file_name):

    storage_client = storage.Client()
    """GCS에서 파일 다운로드"""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def process(input_path, seg_image):
    
    # GCS 버킷 및 모델 파일 정보
    model_path = "/tmp/best.pt"  # 로컬 임시 파일로 저장할 경로
    bucket_name = "inisw04-buckit"
    blob_name = "best.pt"
    

    # 임시 디렉토리에 모델 파일 다운로드
    download_blob(bucket_name, blob_name, model_path)
    print("yolo model is ready")

    model = YOLO(model_path)
    output = model(input_path)
    
    if len(output) == 0:
        return None, [], None, "Nothing detected in OD"

    
    od_result = extract_labels(output, model.names)
    od_result = [label for label, _ in od_result]
    
    final_image = visualize_detections(seg_image, output, model.names)
    full_od_result = output
    
    return final_image, od_result, full_od_result, None