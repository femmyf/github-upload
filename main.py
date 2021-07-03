import librosa
import librosa.feature
from librosa.feature.spectral import mfcc
import numpy as np
import csv
import serial
import time

#extract Suara
def extract_features_voice(f):
    y, sr = librosa.load(f)
    mfcc = librosa.feature.mfcc(y)
    
    #print(mfcc)
    with open('tutup.csv','w',newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(['Features'])
        thewriter.writerow(mfcc)
extract_features_voice('Tutup.wav')


    











