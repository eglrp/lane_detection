import numpy as np
import cv2
import os
#import load
root = 'drive/left/'
saveroot='train/'
width = 750
height = 180


def getpitch(img):
	img = img[195:,100:850]
	return img
def saveimg(filename):
	img = cv2.imread(root+filename)
	img = getpitch(img)
	cv2.imwrite(saveroot+filename,img)
def getFileList():
    #i = 0
    path ='drive/left/'
    for dirpath,dirnames,filenames in os.walk(path):
        for file in filenames:
            saveimg(file)
def getpoint(filename):
	img = cv2.imread(saveroot+filename)
	#print(np.shape(img))
	#for x in range(width):
	#	for y in range(height):
	#		if img[y][x][1] == 255 and img[y][x][1]==0:
	#			print(img[y][x])
	print(img[112][200])
if __name__ == '__main__':
	filename='0000000031.png'
	getpoint(filename)
	#saveimg(filename)
	#getlane(filename)
	#getFileList()