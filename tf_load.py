import numpy as np
import cv2
import os
import numpy as np
import random  
def gettraindataset():
	trainlist = []
	path ='nn/y/'
	for dirpath,dirnames,filenames in os.walk(path):
		for file in filenames:
			data = []
			img = cv2.imread(path+file)
			img = img.reshape(30*30,3)
			#img = img / 128 - 1
			data.append(img)
			data.append(1)
			trainlist.append(data)
	path ='nn/n/'
	for dirpath,dirnames,filenames in os.walk(path):
		for file in filenames:
			data = []
			img = cv2.imread(path+file)
			img = img.reshape(30*30,3)
			#img = img / 128 - 1
			data.append(img)
			data.append(0)
			trainlist.append(data)
	random.shuffle(trainlist)
	train_data = []
	train_label = []
	for data in trainlist:
		train_data.append(data[0])
		train_label.append(data[1])
	return train_data,train_label
if __name__ == '__main__':
	train_data,train_label = gettraindataset()