import cv2
#print(cv2.__version__)
import numpy as np
import torch
import  cv2 
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torchvision import datasets 
from torch.utils.data import DataLoader 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

thres = 0.50  # Threshold to detect object
nms_threshold = 0.30
cap = cv2.VideoCapture(1)#('/home/mllab/Downloads/bean.mp4')
#cap.set(3,1280)
#cap.set(4,720)
#cap.set(10,150)
#cap.set(3,640)
#cap.set(4,480)
classNames = []
classFile = "coco.names"
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    #print(classNames)
configPath = "D:\ECE 2nd year sem1\AIML Project\ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
weightsPath = "D:\ECE 2nd year sem1\AIML Project\frozen_inference_graph.pb"
#print(weightsPath)
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        bbox = list(bbox)
        print(classIds,bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))
        # print(type(confs[0]))
        # print(confs)
        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
        print(indices)
        if len(classIds)!=0:
          for classId,confidence,box in zip(classIds, confs, bbox):
            # for i in indices:
            #     i = i[0]
            #     box = bbox[1]
            # x, y, w, h = box[0], box[1], box[2], box[3]
            # cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
            # cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
            #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Output", img)
        cv2.waitKey(10000)
