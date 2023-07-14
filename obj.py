import cv2
import numpy as np
thres = 0.45  # Threshold to detect object
img=cv2.imread("D:\ECE 2nd year sem1\AIML Project\download.jpeg")
#print(img.shape)
#width = 320

up_width = 600
up_height = 400
up_points = (up_width, up_height)
#img = cv2.resize(img, up_points, interpolation=cv2.INTER_LINEAR)
cv2.imshow('Resized Up image by defining height and width', img)
cv2.waitKey()
cv2.destroyAllWindows()
#cv2.resize()
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
#cap.set(10, 70)
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
# while True:
# success, img = cap.read()//
classIds, confs, bbox = net.detect(img, confThreshold=thres)
print(classIds)
print(classIds, bbox)
confs = list(np.array(confs).reshape(1, -1)[0])
# classIds = [item for sublist in classIds for item in sublist]
# confs = [item for sublist in confs for item in sublist]
if len(classIds) != 0:
    for classId, confidence, box in zip(classIds, confs, bbox):
        # for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        
cv2.imshow("Output", img)
cv2.waitKey(10000)
