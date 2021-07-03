import librosa
import librosa.feature
import numpy as np
import pandas as pd
import csv
from csv import reader 
import serial
import time
from scipy.io.wavfile import write 
import sounddevice



#LVQ Function
def lvq_fit(train, target, learn_rate,b,max_epoch):
    label, train_idx = np.unique(target, return_index=True)
    weight = train[train_idx]
    train = np.array([e for i, e in enumerate(zip(train,target)) if i not in train_idx])
    train, target = train[:,0], train[:,1]
    epoch = 0

    while epoch < max_epoch:
        for i, x in enumerate(train):
            distance = [sum((w-x)**2)for w in weight]
            min = np.argmin(distance)
            sign = 1 if target[1] == label[min] else -1
            weight[min] += sign * learn_rate * (x - weight[min])

        learn_rate *= b
        epoch += 1
    return weight, label
# Lvq Predict Function
def lvq_predict(x, weight):
    weight, label = weight
    d = [sum((w-x)**2) for w in weight]
    return label[np.argmin(d)]

# Load DataBase
train = list(reader('DataBase.csv'))
target = [('1','2')]
weight =lvq_fit
# Record Voice
def voice_input():
    fs = 44100
    second = 3
    print('Recording......')
    record_voice = sounddevice.rec(int(second*fs),samplerate=fs,channels=2)
    sounddevice.wait()
    write('output.wav',fs,record_voice)
#Ekstrak Voice Inputan
y = (voice_input())
mfcc = librosa.feature.mfcc(y)

output = lvq_predict(mfcc,weight)

print(output)

#Komunikasi serial dengan arduino
serialcomm = serial.Serial('COM3',9600) #masukan com yang terhubung dengan arduino
serialcomm.timeout = 1

while True:
    i = voice_input
    if i == 'done':
        print('Finished Program')
        break
    serialcomm.write(i.encode())
    time.sleep(0.5)
    print(serialcomm.readline().decode('ascii'))
serialcomm.close()