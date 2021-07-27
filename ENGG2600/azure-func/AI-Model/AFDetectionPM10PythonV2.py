# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:56:29 2021

@author: Reza
"""

import numpy as np
from tensorflow.keras.models import load_model
from scipy.signal import butter, lfilter, resample
from scipy import signal
from scipy.signal import convolve as sig_convolve
## loading the model
# Set this to the model's path
Model = load_model('AI-Model/NewAIModel.h5')

## Reading dat file
# Set this path
path_dat = 'AI-Model/data/0001-1/ecg.dat'

fin = open(path_dat, 'rb')
a = np.fromfile(fin, dtype=np.dtype('<u2'))
X=[]
for x in a:
  #  print(x)
    X.append(x)
new_sample = np.asarray(X)


## Filtering

from scipy import signal
from scipy.signal import convolve as sig_convolve
temp1 = signal.medfilt(new_sample, kernel_size=np.int64(250/5+1))
temp2 = signal.medfilt(temp1, kernel_size=np.int64(2*250/3+1))
new_sample_subt = new_sample-temp2
b = signal.firwin(21,  20/250 , window='hamming')
new_sample_flt = sig_convolve(np.expand_dims(new_sample_subt,axis=0), b[np.newaxis, :], mode='valid')


# Normalisation
def Normalize(data):
    maxValue = np.max(data)
    minValue = np.min(data)
    temp = (data-minValue)/(maxValue-minValue)
    return temp
new_sample_flt_norm = np.squeeze(np.transpose(Normalize(new_sample_flt)))


# Cutting and Padding
def embeddingTupel(data,maxLength):
    maxLength = np.int(maxLength)
    temp = np.full((maxLength-data.shape[0],),0)
    dataEmb = np.concatenate((data,temp.reshape(temp.shape[0],)),axis=0)
    return dataEmb

if new_sample_flt_norm.shape[0]>7500:
    new_sample_flt_norm_resize = new_sample_flt_norm[0:7500,:]
else: 
    new_sample_flt_norm_resize = embeddingTupel((new_sample_flt_norm),7500)
    
    
# Upsampling
new_sample_flt_norm_resize_upsamp = resample(new_sample_flt_norm_resize, 9000)

## This is the new line to pad the samples to 60 sec
new_sample_flt_norm_resize_upsamp_padd = embeddingTupel(new_sample_flt_norm_resize_upsamp,18000)

predProb = Model.predict(np.expand_dims(np.expand_dims(new_sample_flt_norm_resize_upsamp_padd,axis=1),axis=0))
print(predProb)
print(predProb.shape)
Prediction = np.argmax(predProb)+1
print(Prediction, "is the prediction of the sample")
def second_largest(array):
    print(array, "array before sorting")
    sortedgivenArray = sorted(array, reverse = True)
    print(array, "array after sorting")
    secondLargestNumber = sortedgivenArray[1]
    return secondLargestNumber
Reject = 0
if np.max(predProb)<0.97:
    print("met the first requirement")
    if (np.max(predProb)-second_largest(predProb))<0.95:
        print("about to change the rejection value to be true")
        Reject = 1
else:
    print("value is not less than 0.97")
print(Prediction)
print(Reject)