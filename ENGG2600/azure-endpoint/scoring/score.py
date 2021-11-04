import json
import numpy as np 
import logging
import os
from tensorflow.keras.models import load_model
from azureml.core.model import Model
from scipy import signal
from scipy.signal import butter, lfilter, resample
from scipy.signal import convolve as sig_convolve

def init():
    global model
    #model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.h5")
    model = load_model("model.h5")

def filter(input):
    temp1 = signal.medfilt(input, kernel_size=np.int64(250/5+1))
    temp2 = signal.medfilt(temp1, kernel_size=np.int64(2*250/3+1))
    new_sample_subt = input-temp2
    b = signal.firwin(21,  20/250, window='hamming')
    new_sample_flt = sig_convolve(np.expand_dims(
        new_sample_subt, axis=0), b[np.newaxis, :], mode='valid')
    return new_sample_flt

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def Normalize(data):
    maxValue = np.max(data)
    minValue = np.min(data)
    temp = (data-minValue)/(maxValue-minValue)
    return temp


def embeddingTupel(data, maxLength):  # Cutting and Padding
    maxLength = np.int(maxLength)
    temp = np.full((maxLength-data.shape[0],), 0)
    dataEmb = np.concatenate((data, temp.reshape(temp.shape[0],)), axis=0)
    return dataEmb


def second_largest(array):
    sortedgivenArray = sorted(array, reverse=True)
    secondLargestNumber = sortedgivenArray[1]
    return secondLargestNumber


def process(data: np.ndarray):
    slice = data[0:2]
    logging.info(data.size)  # for debugging, print data length
    logging.info(slice)  # print first two data points

    # Filter
    new_sample_flt = filter(data)

    # Normalisation
    new_sample_flt_norm = np.squeeze(np.transpose(Normalize(new_sample_flt)))

    if new_sample_flt_norm.shape[0] > 7500:
        new_sample_flt_norm_resize = new_sample_flt_norm[0:7500, :]
    else:
        new_sample_flt_norm_resize = embeddingTupel(
            (new_sample_flt_norm), 7500)

    # Upsampling
    new_sample_flt_norm_resize_upsamp = resample(
        new_sample_flt_norm_resize, 9000)

    # This is the new line to pad the samples to 60 sec
    new_sample_flt_norm_resize_upsamp_padd = embeddingTupel(
        new_sample_flt_norm_resize_upsamp, 18000)

    return np.expand_dims(np.expand_dims(new_sample_flt_norm_resize_upsamp_padd, axis=1), axis=0)

def run(request):
    try:
        data = json.loads(request)
        ecg_array = np.array(data['data'])
        processed = process(ecg_array)
        result = model.predict(processed)
        prediction = np.argmax(result)+1
        reject = 0
        if np.max(result[0]) < 0.97 and (np.max(result[0])-second_largest(result[0])) < 0.95:
            reject = 1
        return (int(prediction), int(reject))
    except Exception as e:
        error = str(e)
        return {'data' : error, "message" : 'unable to classify sample'}

