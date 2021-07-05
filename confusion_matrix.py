import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,recall_score,f1_score
import sys


k=5
########## KNN CODE ############
def distance(v1, v2):
    # Eucledian 
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test):
    dist = []
    
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]
    
    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]
################################

skip = 0
dataset_path = './pca_val_data/'

face_data = [] 
labels = []     # we will use roll number as labels

# class_id = 0 # Labels for the given file
names = {} #Mapping btw rollno - name

# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        #Create a mapping btw class_id and name
        rollNo = fx[:-4]
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        #Create Labels for the class
        target = np.array([rollNo for _ in range(data_item.shape[0])],dtype=object)
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))
testset = np.concatenate((face_dataset,face_labels),axis=1)

def knn_accuracy():
    total_count = testset.shape[0]
    pred_out = []
    actual_out = []
    for (i,row) in enumerate(testset):
        test_face=row[:-1]
        pred = knn(testset,test_face)
        actual_out.append(row[-1])
        pred_out.append(pred)
    actual_out = np.array(actual_out)
    pred_out = np.array(pred_out)
    return actual_out , pred_out
    

actual , predicted = knn_accuracy()
sys.stdout = open('pca_result', 'a')
print('For k = '+ str(k))
print()
cm = confusion_matrix(actual,predicted,labels = np.unique(actual))
print('*********CONFUSION_MATRIX**********')
print()
print(cm)
print('Accuracy : {:.2f}' .format(accuracy_score(actual,predicted)))
print('Recall : {:.2f}' .format(recall_score(actual,predicted,average='macro')))
print('f1 Score : {:.2f}' .format(f1_score(actual,predicted , zero_division=1,average='macro')))
print()
print('***********REPORT*************')
print(classification_report(actual,predicted , zero_division=1))
print()