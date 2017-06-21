import numpy as np
import cv2
img = cv2.imread('E:/code/lane_detection/drive/left/0000000009.png')
img[:195,:,1]=255
img[:,:100,1]=255
img[:,850:,1]=255
#line(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
for i in range(7):
	cv2.line(img, (100,195+30*i), (850,195+30*i), (0,0,255), thickness=1)
for i in range(26):
	cv2.line(img, (100+30*i,195), (100+30*i,375), (0,0,255), thickness=1)
cv2.imwrite('messigray.png',img)
#cv2.imshow('s',img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
	cv2.destroyAllWindows()
#for i in range(5):
#	cv2.line(img, (0,end_point_h+39*i), (900,end_point_h+39*i), (0,0,255), thickness=1)
#for i in range(31):
#	cv2.line(img, (0+i*30,end_point_h), (0+i*30,375), (0,0,255), thickness=1)
#pitch = img[0:30,0:30,:]
#cv2.imshow('image',pitch)
#cv2.imwrite('F:/lane_detection/nn/messigray.png',pitch)
#k = cv2.waitKey(0)
#if k == 27:         # wait for ESC key to exit
#    cv2.destroyAllWindows()
#elif k == ord('s'): # wait for 's' key to save and exit
#    cv2.imwrite('messigray.png',pitch)
#    cv2.destroyAllWindows()