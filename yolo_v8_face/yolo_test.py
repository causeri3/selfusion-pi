""" Script is not needed, just here if you want to test the selfie functionality separated."""

import os
import sys

# pretend you are in the directory above
try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
project_dir = os.path.abspath(os.path.join(os.path.dirname(this_file), '..'))
sys.path.insert(0, project_dir)

import logging
from yolo_v8_face.utils.video import Stream
import cv2
from datetime import datetime


logging.basicConfig(encoding='utf-8', level=logging.DEBUG)


stream = Stream(
    #available_devices = device, # [0],
    see_detection=True,
    available_devices=None,
    face_size_threshold=0.05
    )

selfie = stream.draw_boxes()

# window_name = 'SELFIE'
#cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
#cv2.imshow(window_name, selfie)
# cv2.waitKey(1)
# cv2.destroyWindow(window_name)
# cv2.waitKey(1)

stamp = datetime.now().strftime("%-d_%B_%H_%M").lower()
cv2.imwrite("yolo_v8_face/images/selfie" + '_' + stamp + ".png", selfie)