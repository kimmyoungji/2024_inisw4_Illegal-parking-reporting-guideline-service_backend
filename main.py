import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import segmentations as seg
import license_plate as lp
import yolo_od as od
import area as ar
import base64
import io

app = Flask(__name__)

def pil_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def main(input_path):

    # SEG MODEL
    try:
        seg_image, areas, car_bbox, seg_result, seg_error = seg.process(input_path)

        if seg_error:
            return None, None, None, None, None, seg_error
    except Exception as e:
        return None, None, None, None, None, str(e) + " (in SEG)"

    # AREA MODEL
    try:
        area_output = ar.process(areas)
    except Exception as e:
        return seg_image, seg_result, None, None, None, str(e) + " (in AREA)"

    # LP MODEL
    try:
        seg_lp_image, license_number, lp_error = lp.process(input_path, car_bbox, seg_image)

        if lp_error:
            return seg_image, seg_result, area_output, None, None, lp_error  + " (in LP)"

    except Exception as e:
        return seg_image, seg_result, area_output, None, None, str(e)  + " (in LP)"

    # OD MODEL
    try:
        final_image, od_result, full_od_result, od_error = od.process(input_path, seg_image)
        
        if od_error:
            return seg_lp_image, seg_result, area_output, license_number, None, od_error + " (in OD)"
        
    except Exception as e:
        return seg_lp_image, seg_result, area_output, license_number, None, str(e) + " (in OD)"
    
    
    
    # result
    return final_image, seg_result + od_result, area_output, license_number, full_od_result, None


@app.route('/process_image', methods=['POST'])
def process_image_function(request):
    # input image
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(image_file.filename)
    input_path = f"/tmp/{filename}"
    image_file.save(input_path)
    
    # main process
    final_image, od_result, area_output, license_number, full_od_result, error = main(input_path)
    
    response_data = {
        'msg': error if error else "200 OK",
        'image': pil_image_to_base64(final_image) if final_image else None,
        'od_result': od_result if od_result else [],
        'area': area_output if area_output else {},
        'license_number': license_number if license_number else ""
    }
    
    # output
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(debug=True)
