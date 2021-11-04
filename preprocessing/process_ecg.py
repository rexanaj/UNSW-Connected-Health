import logging
import numpy as np
from tensorflow.keras.models import load_model
from scipy import signal
from scipy.signal import butter, lfilter, resample
from scipy.signal import convolve as sig_convolve

import time, datetime

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO: Change this to the model's path 
Model = load_model('AIModel.h5')


def filter(input):
    """
    Filters the sample using a custom filter
    Returns the filtered sample 
    """
    temp1 = signal.medfilt(input, kernel_size=np.int64(250/5+1))
    temp2 = signal.medfilt(temp1, kernel_size=np.int64(2*250/3+1))
    new_sample_subt = input-temp2
    b = signal.firwin(21,  20/250, window='hamming')
    new_sample_flt = sig_convolve(np.expand_dims(
        new_sample_subt, axis=0), b[np.newaxis, :], mode='valid')
    return new_sample_flt


def butter_bandpass(lowcut, highcut, fs, order):
    """
    Designs a bandpass filter to use on the sample 
    Returns the filter coefficients 
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """
    Filers the sample using the bandpass filter
    Returns the filtered sample 
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def normalise(data):
    """
    Returns the normalised sample
    """
    maxValue = np.max(data)
    minValue = np.min(data)
    temp = (data-minValue)/(maxValue-minValue)
    return temp


def embed_tuple(data, maxLength):
    """
    Cuts and pads the sample
    """
    maxLength = np.int(maxLength)
    temp = np.full((maxLength-data.shape[0],), 0)
    dataEmb = np.concatenate((data, temp.reshape(temp.shape[0],)), axis=0)
    return dataEmb


def second_largest(array):
    """
    Returns the second-largest element in an array  
    """
    sortedgivenArray = sorted(array, reverse=True)
    secondLargestNumber = sortedgivenArray[1]
    return secondLargestNumber


def process(data:np.ndarray):
    """
    Pre-processes the sample
    """
    slice = data[0:2]
    logging.info(data.size)  # for debugging, print data length
    logging.info(slice)  # print first two data points

    # Filter
    new_sample_flt = filter(data)

    # Normalisation
    new_sample_flt_norm = np.squeeze(np.transpose(normalise(new_sample_flt)))

    if new_sample_flt_norm.shape[0] > 7500:
        new_sample_flt_norm_resize = new_sample_flt_norm[0:7500, :]
    else:
        new_sample_flt_norm_resize = embed_tuple(
            (new_sample_flt_norm), 7500)

    # Upsampling
    new_sample_flt_norm_resize_upsamp = resample(
        new_sample_flt_norm_resize, 9000)

    # This is the new line to pad the samples to 60 sec
    new_sample_flt_norm_resize_upsamp_padd = embed_tuple(
        new_sample_flt_norm_resize_upsamp, 18000)

    predProb = Model.predict(np.expand_dims(np.expand_dims(
        new_sample_flt_norm_resize_upsamp_padd, axis=1), axis=0))

    Prediction = np.argmax(predProb)+1
    Reject = 0
    if np.max(predProb[0]) < 0.97 and (np.max(predProb[0])-second_largest(predProb[0])) < 0.95:
        Reject = 1

    return (int(Prediction), int(Reject), model_name)

if __name__ == '__main__':
    # TODO: Change this line to read a sample .dat file 
    f = open('0001-12.dat', 'rb')
    a = np.frombuffer(f.read(), dtype=np.dtype('<u2'))
    X = []
    for x in a:
        X.append(x)
    ecg_array = np.asarray(X)

    (prediction, reject, model) = process(ecg_array)
