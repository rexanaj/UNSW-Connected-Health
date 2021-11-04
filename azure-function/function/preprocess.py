import numpy as np
import matplotlib.pyplot as plot
from scipy.signal import resample, convolve, firwin, medfilt
from sklearn import preprocessing

"""
An older preprocess file - refer to preprocess_ecg.py in other folders for 
updated version
"""

def plotData(data):
    time = np.linspace(0, 30, 7500)
    plot.plot(time, data)
    plot.show()
    return

def butter_band(sample):
    temp1 = medfilt(sample, kernel_size=np.int64(250/5+1)) 
    temp2 = medfilt(temp1, kernel_size=np.int64(2*250/3+1)) 
    sample_subt = sample - temp2 
    b = firwin(41, 40/250 , window='hamming') 
    sample_flt = convolve(np.expand_dims(sample_subt,axis=0), b[np.newaxis, :], mode='valid')
    return sample_flt

def processData(newSample):
    # plotData(newSample)

    # band pass filtering needs to be done
    filteredSig = butter_band(newSample)
    # plotData(filteredSig)

    #normalise the data
    norm = preprocessing.MinMaxScaler(feature_range=(0,1))
    normData = norm.fit_transform(filteredSig.reshape(-1,1))
    
    #ensure that sample is of correct length
    if (normData.size < 7500):
        padding = np.zeros((7500 - normData.size, 1))
        normData = np.append(normData, padding)
        # normData = np.pad(normData, (7500 - normData.size, 0), 'constant', constant_values=(0))
    elif (normData.size > 7500):
        normData = normData[0 : 7500]
    
    #resample the data to be at 300hz
    resamped = resample(normData, 30 * 300)

    # reshape the sample to dimension of (1, 9000, 1)
    reshaped = resamped.reshape((1,9000,1))

    return reshaped

if __name__ == "__main__":
    processData()
