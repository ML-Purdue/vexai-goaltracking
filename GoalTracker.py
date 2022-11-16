import torch
import cv2
from grip.BlueFilter import BlueGoalFilterCode

cam = cv2.VideoCapture(0)
detector = BlueGoalFilterCode()
model = torch.hub.load('ultralytics/yolov5', 'custom', path='11-10-22.pt', force_reload=True)
model.cpu()

ret = True
while ret:
  ret, frame = cam.read()
  detector.process(frame)
  frame = detector.mask_output
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()