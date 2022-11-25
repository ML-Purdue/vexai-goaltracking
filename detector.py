import pyrealsense2 as rs
import torch
import sys
from grip.BlueFilter import BlueFilter
from grip.RedFilter import RedFilter

pipeline = rs.pipeline()
pipeline.start()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='11-10-22.pt')
model.cuda()

detect_blue = BlueFilter()
detect_red = RedFilter()

def main():
  detect = int(sys.argv[1])
  try:
    while True:
      frame = pipeline.wait_for_frames()
      depth = frame.get_depth_frame()
      if not depth: continue

      if detect == 0:
        frame = detect_blue(frame)
      elif detect == 1:
        frame = detect_red(frame)

      pred = model(frame)

      if not (pred.pandas().xyxy[0].empty):
        inference = pred.pandas().xyxy[0]
        x_mid = (inference.xmin[0] + inference.xmax[0]) / 2 
        y_mid = (inference.ymin[0] + inference.ymax[0]) / 2
        name = inference.name[0]
        dist = depth.get_distance(x_mid, y_mid)
        print("Name: {} Midpoint: {} Dist: {}".format(name, [x_mid, y_mid], dist))
  finally:
    pipeline.stop()

if __name__ == "__main__":
  main()