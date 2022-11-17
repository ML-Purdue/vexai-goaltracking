import pyrealsense2 as rs
import torch
import sys
from grip.BlueFilter import BlueFilter
from grip.RedFilter import RedFilter

pipeline = rs.pipelin()
pipeline.start()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='11-10-22.pt', force_reload=True)
model.cuda()

detect_blue = BlueFilter()
detect_red = RedFilter()

def main():
  detect = int(sys.argv[1])
  try:
    while True:
      frames = pipeline.wait_for_frames()
      depth = frames.get_depth_frame()
      if not depth: continue

      if detect == 0:
        frame = detect_blue(frame)
      elif detect == 1:
        frame = detect_red(frame)

      pred = model(frames)

      print(pred.pandas().xyxy[0])
  finally:
    pipeline.stop()

if __name__ == "__main__":
  main()