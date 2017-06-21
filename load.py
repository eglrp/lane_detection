import numpy as np
import cv2
import os
import numpy as np
import random  
from sklearn import svm
from sklearn.neural_network import MLPClassifier

def showImg():
    filename='drive/left/0000000030.png'
    img = cv2.imread(filename)
    img[180:300,100:850]=[255,10,10]
    cv2.imshow('s',img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('messigray.png',pitch)
        cv2.destroyAllWindows()
def getFileList():
    #for i in os.listdir('drive/left/'):
    i = 0
    path ='drive/left/'
    for dirpath,dirnames,filenames in os.walk(path):
        for file in filenames:
            i = i + 1
            if i%10==0:
                saveRect(file)
def saveRect(filename):
    img = cv2.imread('drive/left/'+filename)
    left = 100
    top = 195
    size = 30
    for x in range(25):
        for y in range(6):
            pitch = img[top+y*size:top+(y+1)*size, left+x*size:left+(x+1)*size]
            pitchname = filename[:-4]+'_'+str(x)+'_'+str(y)+'.png'
            cv2.imwrite('nn/'+pitchname,pitch)
def rename():
    path ='nn/no/'
    for dirpath,dirnames,filenames in os.walk(path):
        for file in filenames:
            pitch = cv2.imread(path + str(file))
            file = file[:-8]
            cv2.imwrite('nn/n/'+file+'.png',pitch)
def getdataset():
    pass
def pretreatment():
    datalist = []
    path ='nn/y/'
    for dirpath,dirnames,filenames in os.walk(path):
        for file in filenames:
            data = []
            img = cv2.imread(path+file)
            img = img.reshape(30*30*3)
            img = img / 128 - 1
            data.append(img)
            data.append(1)
            datalist.append(data)
    path ='nn/n/'
    for dirpath,dirnames,filenames in os.walk(path):
        for file in filenames:
            data = []
            img = cv2.imread(path+file)
            img = img.reshape(30*30*3)
            img = img / 128 - 1
            data.append(img)
            data.append(0)
            datalist.append(data)
    random.shuffle(datalist)
    random.shuffle(datalist)
    #print(np.shape(datalist))
    train_data = []
    train_label = []
    for i in datalist:
        train_data.append(i[0])
        train_label.append(i[1])
    return train_data,train_label

def pretreatmentTest():
    datalist = []
    path ='nn/y_test/'
    for dirpath,dirnames,filenames in os.walk(path):
        for file in filenames:
            data = []
            img = cv2.imread(path+file)
            img = img.reshape(30*30*3)
            img = img / 128 - 1
            data.append(img)
            data.append(1)
            datalist.append(data)
    path ='nn/n_test/'
    for dirpath,dirnames,filenames in os.walk(path):
        for file in filenames:
            data = []
            img = cv2.imread(path+file)
            img = img.reshape(30*30*3)
            img = img / 128 - 1
            data.append(img)
            data.append(0)
            datalist.append(data)
    random.shuffle(datalist)
    random.shuffle(datalist)
    #print(np.shape(datalist))
    test_data = []
    test_label = []
    for i in datalist:
        test_data.append(i[0])
        test_label.append(i[1])
    return test_data,test_label

def trainSVM(train_data, train_label):
    clf = svm.SVC(kernel='linear')
    clf.fit(train_data, train_label)

    return clf

def trainNN(train_data, train_label):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,150,2), random_state=1)
    clf.fit(train_data, train_label)
    return clf

def test(datasets, label, clf):
    prediction = []
    for data in datasets:
        p = clf.predict([data])
        prediction.append(p[0])
        #print(p[0])
    total = 0
    wrong = 0
    for label in label:
        if label!=prediction[total]:
            wrong = wrong + 1
        total = total + 1
    print('the accuracy is :' + str(100 - (wrong / total)*100) + '%')
def distinct(d1,d2):
    d = d1-d2
    d = d**2
    return np.sum(d)

train_data,train_label = pretreatment()
print('训练集读取完成')
test_data,test_label = pretreatmentTest()
print('测试集读取完成')
clf = trainNN(train_data, train_label)
print('神经网络训练完成')
print('再训练集中')
test(train_data,train_label, clf)
print('再测试集中')
test(test_data,test_label, clf)

if __name__ == '__main__':
    pass
    #getFileList()
    