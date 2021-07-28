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

def getPredictionProbability(model, dataset):
    # Process sample
    processedSample = processData(dataset)
    predProb = model.predict(np.expand_dims(np.expand_dims(processedSample,axis=1),axis=0))
    print(predProb, "is the prediction probabilities")
    return predProb

def getPrediction(predictionProbability):
    # Process sample
    prediction = np.argmax(predictionProbability)+1

    # Print results
    if prediction == 1:
        return "AFIB"
    elif prediction == 2:
        return "NSR"
    elif prediction == 3:
        return "Other"
    elif prediction == 4:
        return "Too noisy"
    else:
        return "Model failed"

def isRejected(predictionProbability):
    if np.max(predictionProbability)<0.97 and (np.max(predictionProbability)-second_largest(predictionProbability))<0.95:
        return False
    return True

def processData(dataset):
    filtered = filterData(dataset)
    normalised = np.squeeze(np.transpose(Normalize(filtered)))
    resized = resize(normalised)
    upsampled = resample(resized, 9000)
    reshaped = embeddingTuple(upsampled, 18000)
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
        resized_data = embeddingTuple((dataset),7500)
    return resized_data

def embeddingTuple(data, maxLength):
    maxLength = np.int(maxLength)
    temp = np.full((maxLength-data.shape[0],),0)
    dataEmb = np.concatenate((data,temp.reshape(temp.shape[0],)),axis=0)
    return dataEmb

def second_largest(array):
    sortedgivenArray = sorted(array[0], reverse = True)
    if len(sortedgivenArray) == 0: 
        return 0
    else:
        secondLargestNumber = sortedgivenArray[1]

    return secondLargestNumber
