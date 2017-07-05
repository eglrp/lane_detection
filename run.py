import numpy as np
import cv2
import math
import datetime
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
# 敏感区域范围【195:,100:850】
# 分类结果均保存在mat矩阵中，包括mat矩阵和对应的图片方便找人工(zao)特征(jia)
# 对图片进行切块，并分类，返回mat矩阵和仅保留车道线图像块的图片
# file 文件名，cls训练得到的分类器
def classify(filename, clf):
	img = cv2.imread('drive/left/'+filename+'.png')
	img = img[195:,100:850]
	mat = np.zeros([6,25], np.int8)
	for x in range(25):
		for y in range(6):
			pitch = img[y*30:(y+1)*30, x*30:(x+1)*30]
			pitch = pitch.reshape(30*30*3)
			pitch = pitch / 128 - 1
			p = clf.predict([pitch])
			if p[0]==0:
				img[y*30:(y+1)*30, x*30:(x+1)*30] = [0,0,0]
			else:
				mat[y][x] = 1
	#plt.imshow(img)
	#plt.show()
	return mat,img
	
# 使用霍夫变换对MAT矩阵进行处理，得到直方图和每个角度穿过的图像块信息
def houghtransport(mat):
	epy = 0
	epx = 16*30+15
	bg_black = np.zeros((180,750),np.uint8)
	list = []
	# 先把那个直线搞出来，利用直线判断是否穿过
	for i in range(20,170,2):
		# 绘制直线
		bg_black = np.zeros((180,750),np.uint8)
		if i!=90:
			k = math.tan(i*math.pi/180)
			cv2.line(bg_black,(epx,epy),( int((180/k)+epx),180),255,1)
		else:
			cv2.line(bg_black,(epx,epy),(epx,180),255,1)
		# 获得穿过的数据
		
		sum = 0 
		poslist = []
		for x in range(25):
			for y in range(1,6):
				if mat[y][x] == 1:
					size = 30
					pitch = bg_black[y*size:(y+1)*size, x*size:(x+1)*size]
					if 255 in pitch:
						poslist.append([x,y])
						sum = sum + 1
		list.append([sum, poslist])
	return list
# 获取第一条车道线的信息，拟合直线
def getline(list,mat):
	result = []
	# 得到一个空的背景
	bg_black = np.zeros((375,1242),np.uint8)
	# 获取最长的那条线
	max = 0
	# l里面保存了最多的块块，存的列表在直方图中的id，l里面是直方图里面的索引
	l = []
	i = 0
	for info in list:
		if info[0] > max:
			max = info[0]
			l = [i]
		elif info[0] == max:
			l.append(i)
		i = i + 1
	if len(l) == 1:
		# 妥妥的就是这条了，不过还是先跳过吧
		#result = [list[l[0]]]
		result = list[l[0]][1]
	elif len(l) > 1:
		# 需要算法判断一下,具体算法见shiyan.py
		result = cluster(l)
		# 然后在result里面把最重要的给提取出来
		result = getlanerexlist(result, list)
		#print(result)
	return result
	
	
# 获取第二条车道线信息
def getsecondline(angle,reclist):
	pass
def cluster(l):
	result = []
	i = 0
	while i < len(l):
		ele = []
		step = 0
		while (l[i]+step) in l:
			ele.append(l[i+step])
			step = step + 1
		# 这里把ele插入返回的列表中
		result.append(ele)
		i = i + step
	return result
	
def getlanerexlist(result,list):
	max = 0
	index = 0
	for i in range(len(result)):
		if len(result[i]) > max:
			max = len(result[i])
			index = i
	if max%2 == 0:
		# 偶数，已完成
		info = result[index]
		n1 = int(len(info)/2) - 1
		n2 = int(len(info)/2)
		n1 = info[n1]
		n2 = info[n2]
		l1 = list[n1][1]
		l2 = list[n2][1]
		result = l1+l2
	else:
		# 奇数，未完成
		print('是奇数')
		info = result[index]
		n = math.floor(len(info)/2)
		n = info[n]
		result = list[n][1]
		#result = [list[n]]
	return result
def showlane(result, filename, action):
	lane = cv2.imread('black_big.png')
	img = cv2.imread('drive/left/'+filename+'.png')
	for p in result:
		x = p[0]
		y = p[1]
		pitch = img[195+y*30:195+(y+1)*30, 100+x*30:100+(x+1)*30]
		lane[195+y*30:195+(y+1)*30, 100+x*30:100+(x+1)*30] = pitch
	if action == True:
		lane = getlineroi(lane)
	return lane
	
def getlineroi(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	l = np.array([0, 0, 180])
	h = np.array([255, 25, 255])
	mask = cv2.inRange(img, l, h)
	l = cv2.bitwise_and(img, img, mask=mask)
	l = cv2.cvtColor(l, cv2.COLOR_HSV2BGR)
	return l
def removelane(mat,result):
	for ele in result:
		x = ele[0]
		y = ele[1]
		mat[y][x] = 0
	return mat
	
if __name__ == '__main__':
	clf = joblib.load("model.m")
	filename = '0000000022'
	starttime = datetime.datetime.now()
	mat,img = classify('0000000022', clf)
	#mat,img2 = classify('0000000031', clf)
	#mat,img3 = classify('0000000051', clf)
	#mat,img4 = classify('0000000079', clf)
	
	#temp1 = np.zeros((375,1242,3),np.uint8)
	#temp1[195:,100:850] = img1
	#img1 = temp1
	list = houghtransport(mat)
	result = getline(list,mat)
	firstlane = showlane(result, filename,True)

	removelane(mat,result)
	list = houghtransport(mat)
	result = getline(list,mat)
	secondlane = showlane(result, filename,True)

	removelane(mat,result)
	list = houghtransport(mat)
	result = getline(list,mat)
	thirdlane = showlane(result, filename,True)
	
	endtime = datetime.datetime.now()
	print ((endtime - starttime))
	
	fig = plt.figure() 
	ax = fig.add_subplot(221)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(firstlane)
	
	ax = fig.add_subplot(222)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(secondlane)
	
	ax = fig.add_subplot(223)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(thirdlane)
	
	lane = firstlane + secondlane + thirdlane
	ax = fig.add_subplot(224)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(lane)

	plt.show()
'''
	list = houghtransport(mat)
	result = getline(list,mat)
	firstlane = showlane(result, filename,True)

	removelane(mat,result)
	list = houghtransport(mat)
	result = getline(list,mat)
	secondlane = showlane(result, filename,True)

	removelane(mat,result)
	list = houghtransport(mat)
	result = getline(list,mat)
	thirdlane = showlane(result, filename,True)
	
	endtime = datetime.datetime.now()
	print ((endtime - starttime))
'''