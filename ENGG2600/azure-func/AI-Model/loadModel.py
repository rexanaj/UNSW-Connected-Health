"""
1 = Atrial Fibrillation
2 = Normal Sinus Rhythm
3 = Other Arrhythmia
4 = Too Noisy
"""
import numpy as np
from scipy.signal import resample
from scipy import signal
from scipy.signal import convolve as sig_convolve

def getPrediction(model, dataset):
    # Process sample
    processedSample = processData(dataset)

    prediction = np.argmax(model.predict(processedSample)) + 1

    # Print results
    if prediction == 1:
        return "AFIB"
    elif prediction == 2:
        return "NSR"
    elif prediction == 3:
        return "other"
    elif prediction == 4:
        return "Too noisy"
    else:
        return "Model failed"

def processData(dataset):
    filtered = filterData(dataset)
    normalised = Normalize(filtered)
    resized = resize(normalised)
    upsampled = resample(resized, 9000)
    reshaped = upsampled.reshape((1,9000,1))
    return reshaped

def filterData(dataset):
    temp1 = signal.medfilt(dataset, kernel_size=np.int64(250/5+1))
    temp2 = signal.medfilt(temp1, kernel_size=np.int64(2*250/3+1))
    dataset_subt = dataset-temp2
    b = signal.firwin(21,  20/250 , window='hamming')
    dataset_flt = sig_convolve(np.expand_dims(dataset_subt,axis=0), b[np.newaxis, :], mode='valid')
    return dataset_flt

def Normalize(data):
    maxValue = np.max(data)
    minValue = np.min(data)
    temp = (data-minValue)/(maxValue-minValue)
    return temp

def resize(dataset):
    if dataset.shape[0]>7500:
        resized_data = dataset[0:7500,:]
    else: 
        padding = np.zeros((7500 - dataset.size, 1))
        resized_data = np.append(dataset, padding)
    return resized_data

def embeddingTupel(data, maxLength):
    maxLength = np.int(maxLength)
    temp = np.full((maxLength-data.shape[0],),0)
    dataEmb = np.concatenate((data,temp.reshape(temp.shape[0],)),axis=0)
    return dataEmb