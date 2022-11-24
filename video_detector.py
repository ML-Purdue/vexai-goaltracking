import cv2
import sys
import os
import numpy as np
from grip.BlueFilter import BlueFilter
from grip.RedFilter import RedFilter
import torch

red_detector = RedFilter()
blue_detector = BlueFilter()
model = torch.hub.load('ultralytics/yolov5', 'custom', path='11-10-22.pt')
model.cuda()

def main():
  model_input = sys.argv[1]
  if (model_input in ['0','1','2','3','4','5','6','7','8','9']):
    model_input = int(model_input)
  cam = cv2.VideoCapture(model_input)
  conf_thres = sys.argv[2]
  model.conf = float(conf_thres)
  detect = int(sys.argv[3]) #0 for blue 1 for red 2 for both


  ret = True
  while ret:
    ret, frame = cam.read()
    
    if detect == 0:
      frame = detect_blue(frame)
    elif detect == 1:
      frame = detect_red(frame)
    else:
      frame = cv2.bitwise_or(detect_red(frame), detect_blue(frame))
    
    pred = model(frame)

    for row in pred.pandas().xyxy[0].iterrows():
        pt1 = (int(row[1]["xmin"]), int(row[1]["ymin"]))
        pt2 = (int(row[1]["xmax"]), int(row[1]["ymax"]))
        color = (255, 0, 0)
        thickness = 2
        frame = cv2.rectangle(frame, pt1, pt2, color, thickness)
      

    if not (pred.pandas().xyxy[0].empty):
      inference = pred.pandas().xyxy[0]
      x_mid = (inference.xmin[0] + inference.xmax[0]) / 2 
      y_mid = (inference.ymin[0] + inference.ymax[0]) / 2
      print([x_mid, y_mid])

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cam.release()
  cv2.destroyAllWindows()

def detect_blue(frame):
  blue_detector.process(frame) #GRIP pipeline
  frame = blue_detector.mask_output
  #frame = mask_red(frame)
  return frame

def detect_red(frame):
  red_detector.process(frame) #GRIP pipeline
  frame = red_detector.mask_output
  #frame = mask_red(frame)
  
  return frame

def mask_red(image):
  result = image.copy()
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
  # lower boundary RED color range values; Hue (0 - 10)
  lower1 = np.array([0, 100, 20])
  upper1 = np.array([10, 255, 255])
  
  # upper boundary RED color range values; Hue (160 - 180)
  lower2 = np.array([160,100,20])
  upper2 = np.array([179,255,255])
  
  lower_mask = cv2.inRange(image, lower1, upper1)
  upper_mask = cv2.inRange(image, lower2, upper2)
  
  full_mask = lower_mask + upper_mask
  
  result = cv2.bitwise_and(result, result, mask=full_mask)
  return result

if __name__ == "__main__":
  main()