import numpy as np
import math
import cv2
import time
import matplotlib.pyplot as plt
def getfirstline(filename,source,angle):
	img = cv2.imread('black_big.png')
	s = cv2.imread('drive/left/0000000022.png')
	mat = np.loadtxt(filename)

if __name__ == '__main__':
	img = getfirstline('mat.txt','0000000022.png',158)
	#cv2.imshow('dd',img)
	#if cv2.waitKey(1) & 0xFF == ord('q'):
	#	cv2.destroyAllWindows()
'''
		cv2.imshow('dd',bg)	
		time.sleep(0.42)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()
'''