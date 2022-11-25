import pyrealsense2 as rs
import torch
import sys
import numpy as np
from grip.RedFilter import RedFilter
from grip.BlueFilter import BlueFilter

import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='11-10-22.pt')
model.cuda()
model.conf = 0.8

blue_detector = BlueFilter()
red_detector = RedFilter()

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

def main():
  detect = int(sys.argv[1])
  try:
    while True:
      frames = pipeline.wait_for_frames()
      frame = np.asanyarray(frames.get_color_frame().get_data())
      depth = frames.get_depth_frame()
      if not depth: continue

      if detect == 0:
        frame = detect_blue(frame)
      elif detect == 1:
        frame = detect_red(frame)

      pred = model(frame)

      #TESTING
      # for row in pred.pandas().xyxy[0].iterrows():
      #   pt1 = (int(row[1]["xmin"]), int(row[1]["ymin"]))
      #   pt2 = (int(row[1]["xmax"]), int(row[1]["ymax"]))
      #   color = (255, 0, 0)
      #   thickness = 2
      #   frame = cv2.rectangle(frame, pt1, pt2, color, thickness)

      if not (pred.pandas().xyxy[0].empty):
        inference = pred.pandas().xyxy[0]
        x_mid = (inference.xmin[0] + inference.xmax[0]) / 2 
        y_mid = (inference.ymin[0] + inference.ymax[0]) / 2
        name = inference.name[0]
        dist = depth.get_distance( int(x_mid), int(y_mid))
        print("Name: {} Midpoint: {} Dist: {}".format(name, [x_mid, y_mid], dist))

      #TESTING ONLY      
      # cv2.imshow('frame', frame)
      # if cv2.waitKey(1) & 0xFF == ord('q'):
      #       break
  finally:
    pipeline.stop()

def detect_blue(frame):
  blue_detector.process(frame) #GRIP pipeline
  return blue_detector.mask_output

def detect_red(frame):
  red_detector.process(frame) #GRIP pipeline
  return red_detector.mask_output

if __name__ == "__main__":
  main()
