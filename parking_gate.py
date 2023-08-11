import cv2
import numpy as np
#from dynamikontrol import Module
import os

# 현재 디렉토리 확인
current_directory = os.getcwd()

# 모델 파일의 절대 경로 생성
model_cfg_path = os.path.join(current_directory, 'cfg', 'yolov4-ANPR.cfg')
model_weights_path = os.path.join(current_directory, 'yolov4-ANPR.weights')

# 모델 파일의 존재 여부 확인
if os.path.exists(model_cfg_path) and os.path.exists(model_weights_path):
    print("모델 파일이 존재합니다.")
else:
    print("모델 파일이 존재하지 않습니다.")

CONFIDENCE = 0.9 # 0~1
THRESHOLD = 0.3 
LABELS = ['Car', 'Plate']
CAR_WIDTH_TRESHOLD = 500 #자동차의 크기 500픽셀 이상일 때

cap = cv2.imread('01.jpg') #cv2.VideoCapture(0) # 웹캠 영상 쓸거면 영상 패스

net = cv2.dnn.readNetFromDarknet('cfg/yolov4-ANPR.cfg', 'yolov4-ANPR.weights')
# 모델 설정값 
#module = Module()

while cap.isOpened():#이미지 열기
    ret, img = cap.read() #읽기
    if not ret:
        break

    H, W, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255., size=(416, 416), swapRB=True)
    net.setInput(blob)
    output = net.forward()

    boxes, confidences, class_ids = [], [], []#네모칸, 컨피던스, 아이디

    for det in output: 
        box = det[:4]#앞 4개 박스 정보 x y w h
        scores = det[5:] # 
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONFIDENCE:#0.9보다 크면
            cx, cy, w, h = box * np.array([W, H, W, H])#픽셀값으로 변환
            x = cx - (w / 2)
            y = cy - (h / 2)

            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]

            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
            cv2.putText(img, text='%s %.2f %d' % (LABELS[class_ids[i]], confidences[i], w), org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

           # if class_ids[i] == 0:#0 = 차 
               # if w > CAR_WIDTH_TRESHOLD:#차의 너비가 크면 모터 제어
                #    module.motor.angle(80)
                #else:
                 #   module.motor.angle(0) #차의 너비가 아니면 닫기
    else:
        #module.motor.angle(0)

        cv2.imshow('result', img)#결과보기
    
        if cv2.waitKey(1) == ord('q'):
            break
