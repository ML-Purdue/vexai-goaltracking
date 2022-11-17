import torch
import cv2
import sys
import os
from grip.BlueFilter import BlueGoalFilterCode
from grip.RedFilter import RedGoalFilterCode

def main():
  model_input = sys.argv[1]
  cam = cv2.VideoCapture(model_input)
  detector = BlueGoalFilterCode()
  model = torch.hub.load('ultralytics/yolov5', 'custom', path='11-10-22.pt', force_reload=True)
  model.conf = 0.5
  model.cpu()

  ret = True
  while ret:
    ret, frame = cam.read()
    detector.process(frame)
    frame = detector.mask_output
    
    pred = model(frame)
    for row in pred.pandas().xyxy[0].iterrows():
        pt1 = (int(row[1]["xmin"]), int(row[1]["ymin"]))
        pt2 = (int(row[1]["xmax"]), int(row[1]["ymax"]))
        color = (255, 0, 0)
        thickness = 2
        frame = cv2.rectangle(frame, pt1, pt2, color, thickness)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cam.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()