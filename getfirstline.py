import numpy as np
import math
import cv2
import time
import matplotlib.pyplot as plt
def getacross(img,mat,bg,s):
	for x in range(25):
		for y in range(1,6):
			if mat[y][x]==1:
				size = 30
				pitch = bg[y*size:(y+1)*size, x*size:(x+1)*size]
				pitch = pitch.reshape(30*30)
				for i in pitch:
					if i!=0:
						#执行到这里说明真是穿过了
						rec = s[195+y*30:195+(y+1)*30, 100+x*30:100+(x+1)*30]
						img[195+y*30:195+(y+1)*30, 100+x*30:100+(x+1)*30] = rec
						break
	return img
def getfirstline(filename,source,angle):
	img = cv2.imread('black_big.png')
	s = cv2.imread('drive/left/'+source)
	mat = np.loadtxt(filename)
	sum = 0
	t = 0
	#提取消失点
	for x in range(25):
		if mat[0][x] == 1:
			sum = sum + x
			t = t + 1
	mid_x = int((sum / t) * 30)
	mid_y = 15
	p1 = (mid_x, mid_y)
	p2 = (mid_x, 180)
	if angle==90:
		pass
	else:
		k = math.tan(math.pi*angle/180)
		p2 = (int((180-mid_y)/k + mid_x), 180)
	bg = cv2.imread('black.png')
	bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
	cv2.line(bg, p1, p2, 255)
	img = getacross(img,mat,bg,s)
	return img
	
if __name__ == '__main__':
	img1 = getfirstline('mat.txt','0000000022.png',158)
	img2 = getfirstline('mat31.txt','0000000031.png',156)
	img3 = getfirstline('mat51.txt','0000000051.png',164)
	img4 = getfirstline('mat79.txt','0000000079.png',158)
	fig = plt.figure() 
	ax = fig.add_subplot(221)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(img1)
	ax = fig.add_subplot(222)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(img2)
	ax = fig.add_subplot(223)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(img3)
	ax = fig.add_subplot(224)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(img4)
	plt.show()
'''
		cv2.imshow('dd',bg)	
		time.sleep(0.42)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()
'''
