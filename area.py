def process(areas):
    total_pixels = sum(item['pixels'] for item in areas)
    road_pixels = 0
    sidewalk_pixels = 0
    max_car_pixels = 0

    # Process each entry
    for item in areas:
        label = item['label']
        pixels = item['pixels']
        
        if label == 'road':
            road_pixels = pixels
        elif label == 'sidewalk':
            sidewalk_pixels = pixels
        elif label == 'car':
            if pixels > max_car_pixels:
                max_car_pixels = pixels

    # Calculate the ratios
    road_ratio = road_pixels / total_pixels
    sidewalk_ratio = sidewalk_pixels / total_pixels
    max_car_ratio = max_car_pixels / total_pixels

    result = {
        "road_ratio": f"{road_ratio:.2%}",
        "sidewalk_ratio": f"{sidewalk_ratio:.2%}",
        "max_car_ratio": f"{max_car_ratio:.2%}"
    }
    
    return result