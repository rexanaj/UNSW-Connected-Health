"""
1 = Atrial Fibrillation
2 = Normal Sinus Rhythm
3 = Other Arrhythmia
4 = Too Noisy
"""

import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from .preProcess import processData

def getPrediction(model, sampleNum, datafile):
    a = np.fromfile(datafile, dtype=np.dtype('<u2'))
    X = []
    for x in a:
        X.append(x)
    newSample = np.asarray(X)

    # Process sample
    processedSample = processData(newSample)

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
