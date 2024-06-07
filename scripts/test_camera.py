import cv2
import pyrealsense2
import numpy as np
from realsense_depth import *

dc = DepthCamera()



while(1):

    contours = []
    ret, depth_frame, color_frame = dc.get_frame()
    img = color_frame
    
    cv2.circle(img,(50,50),5,(0,255,0),-1)

    cv2.imshow('img',img)
    cv2.waitKey(1)
cv2.destroyAllWindows()