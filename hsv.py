import numpy as np
import cv2
import os
import load
# 这里就是获取图像中间的有效区域
def getpitch(img):
	img[:225-30,:]= [0,0,0]
	img[:,:100]=[0,0,0]
	img[:,850:]=[0,0,0]
	return img
size = 30
root = 'F:/lane_detection/drive/left/'
img = cv2.imread('drive/left/0000000079.png')
img = getpitch(img)
clf = load.clf
hmatrix = np.zeros([6,25], np.int8)
for x in range(25):
	for y in range(6):
		pitch = img[195+y*size:195+(y+1)*size, 100+x*size:100+(x+1)*size]
		pitch = pitch.reshape(30*30*3)
		pitch = pitch / 128 - 1
		p = clf.predict([pitch])
		if p==0:
			img[195+y*size:195+(y+1)*size, 100+x*size:100+(x+1)*size]=[0,0,0]
		else:
			hmatrix[y][x] = 1
np.savetxt("mat79.txt",hmatrix)
#cv2.imwrite('d:/0000000022.png',img)
#for i in range(30,150,2)
#	print(i)
#print(hmatrix)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27:
	cv2.destroyAllWindows()
