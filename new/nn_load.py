import cv2
import os
import csv
import numpy as np
train_y_list = []
train_n_list = []
test_data = []
test_label = []
for dirpath,dirnames,filenames in os.walk('nn/y'):
    for filename in filenames:
        img = cv2.imread('nn/y/'+filename)
        #train_y_list.append([img,[0,1]])
        train_y_list.append(img)

for dirpath,dirnames,filenames in os.walk('nn/n'):
    for filename in filenames:
        img = cv2.imread('nn/n/'+filename)
        train_n_list.append(img)

for dirpath,dirnames,filenames in os.walk('nn/n_test'):
    for filename in filenames:
        img = cv2.imread('nn/n_test/'+filename)
        test_data.append(img)
        test_label.append([0,1])

for dirpath,dirnames,filenames in os.walk('nn/y_test'):
    for filename in filenames:
        img = cv2.imread('nn/y_test/'+filename)
        test_data.append(img)
        test_label.append([1,0])



train_y_list = np.array(train_y_list)
train_n_list = np.array(train_n_list)
test_data = np.array(test_data)
test_label = np.array(test_label)
