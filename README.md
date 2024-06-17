# 불법주정차 신고를 위한 가이드라인 제공 서비스의 Serverless backend
Google cloud functions를 활용한 서버리스 백엔드 코드입니다.

main.py의 process_image_function 함수가 진입점입니다.
각 주요 프로세스는 yolo_od.py, segmentations.py, area.py, license_plate.py 내 process 함수로 구현되어 있습니다.
