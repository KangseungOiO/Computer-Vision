from realsense_depth import *
from realsense_camera import *
import cv2
import numpy as np

point = (400, 300)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

# dc = DepthCamera()
rs = RealsenseCamera()

while True:
    ret, depth_frame, color_frame = rs.get_frame()
    # Show distance for a specific point
    cv2.circle(color_frame, point, 4, (0, 0, 255))
    distance = depth_frame[point[1], point[0]]

    cv2.putText(color_frame, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.setMouseCallback("Real_Object_Detection", show_distance)
    cv2.imshow('Real_Object_Detection', color_frame)
    # cv2.imshow("depth frame", depth_frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
