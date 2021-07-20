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
