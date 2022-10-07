import csv
from operator import le
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

def check(result):
   for item in result: 
    if item >= 0.6:
        print('Cancer is Benign ')
    else:
        print('Cancer is Malignant')

#create callback
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy') >= 0.98):
           self.model.stop_training = True

newArr = []

with open('/Users/mac/Documents/work life/programming/machine learning/datasets/breast-cancer-data.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter= ',')
     
    
    for row in reader:
        newArr.append(row)
    

#working on the dataset
secondArr = []

index = 1
while index < len(newArr):
    secondArr.append(newArr[index])
    index += 1



trainingLabelRaw = []
for rows in secondArr:
    trainingLabelRaw.append(rows[1])

numericalTraining = []


for item in trainingLabelRaw:
    if item == 'M':
        numericalTraining.append([0])
    elif item == 'B':
        numericalTraining.append([1])



uncutTrainingData = secondArr
for rows in uncutTrainingData:
    del rows[0:2]

print(len(uncutTrainingData))
trainingData = np.array(uncutTrainingData[0:300]).astype(np.float32)
trainingLabel = np.array(numericalTraining[0:300]).astype(np.float32)
print(len(trainingData))
print(len(trainingLabel))
validationData = np.array(uncutTrainingData[300:420]).astype(np.float32)
validationLabel = np.array(numericalTraining[300:420]).astype(np.float32)
print(len(validationData))
print(len(validationLabel))

testData = uncutTrainingData[420:568]
floatTestData = []
for row in testData:
    rows1 = []
    for column in row:
        rows1.append(float(column))
    floatTestData.append(rows1)


testLabel = numericalTraining[420:569]
print(len(trainingData[0]))

 
#working on the model

callback = mycallback()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (30,)),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainingData, trainingLabel, epochs = 700, validation_data=(validationData, validationLabel) , callbacks = [callback])



test1 = model.predict([floatTestData[4]])
test2 = model.predict([floatTestData[48]])
test3 = model.predict([floatTestData[18]])
test4 = model.predict([floatTestData[102]])
test5 = model.predict([floatTestData[81]])
test6 = model.predict([floatTestData[58]])
test7 = model.predict([floatTestData[100]])
test8 = model.predict([floatTestData[21]])
test9 = model.predict([floatTestData[97]])
test10 = model.predict([floatTestData[67]])


check(test1)
print(testLabel[4])
check(test2)
print(testLabel[48])
check(test3)
print(testLabel[18])
check(test4)
print(testLabel[102])
check(test5)
print(testLabel[81])
check(test6)
print(testLabel[58])
check(test7)
print(testLabel[100])
check(test8)
print(testLabel[21])
check(test9)
print(testLabel[97])
check(test10)
print(testLabel[67])


