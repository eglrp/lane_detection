import numpy as np
import math
import cv2
import time
import matplotlib.pyplot as plt
def getcross(mat,bg):
	num = 0;
	for x in range(25):
		for y in range(1,6):
			if mat[y][x]==1:
				size = 30
				pitch = bg[y*size:(y+1)*size, x*size:(x+1)*size]
				pitch = pitch.reshape(30*30)
				for i in pitch:
					if i!=0:
						num = num + 1
						break
			
	return num
def draw(filename,marker,i,linestyle,title):
	plt.subplot(i)
	h_m = np.loadtxt(filename)
	sum = 0
	t = 0
	#提取消失点
	for x in range(25):
		if h_m[0][x] == 1:
			sum = sum + x
			t = t + 1
	mid_x = int((sum / t) * 30)
	mid_y = 15
	p1 = (mid_x, mid_y)
	p2 = (mid_x, 180)
	# 生成直线,i就代表和水平方向的角度
	max = 0
	index = 0
	x = []
	y = []
	for i in range(10,171,2):
		p2 = (mid_x, 180)
		if i != 90:
			k = math.tan(math.pi*i/180)
			p2 = (int((180-mid_y)/k + mid_x), 180)
			#print(p2)
		bg = cv2.imread('black.png')
		bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
		cv2.line(bg, p1, p2, 255)
		num = getcross(h_m,bg)
		if num > max: 
			max = num
			index = i
		#print(num)
		x.append(i)
		y.append(num)
	plt.plot(x,y,linewidth=1, color='r',marker=marker, linestyle=linestyle, markerfacecolor='blue',markersize=5)
	#plt.set_yticks(range(15))
	#plt.xlabel('angle')
	#plt.ylabel('the number of imagepitch line crossed')
	plt.title(title)
	print(index)
if __name__ == '__main__':
	draw('mat.txt','o',221,'-','the 22th frame')
	draw('mat31.txt','o',222,'-','the 31th frame')
	draw('mat51.txt','o',223,'-','the 51th frame')
	draw('mat79.txt','o',224,'-','the 79th frame')
	#draw('mat.txt','o',221,'-')
	#draw('mat31.txt','s',222,'--')
	#draw('mat51.txt','*',223,':')
	#draw('mat79.txt','v',224,'-.')
	plt.legend()
	plt.show() 
'''
		cv2.imshow('dd',bg)	
		time.sleep(0.42)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()
'''