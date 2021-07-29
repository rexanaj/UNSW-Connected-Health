from io import open_code
from json.decoder import JSONDecodeError
import logging
import sys

from tensorflow.python.platform.tf_logging import log_first_n
import azure.functions as func
import numpy as np
from .processPrediction import getPredictionProbability, getPrediction, isRejected
from tensorflow.keras.models import load_model
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    sampleData = None
    try:
        rawData = req.get_json()
        sampleData = np.asarray(rawData)
    except ValueError:
        logging.info("ValueError for Raw Data")
        pass
    if (sampleData is None):
        if req.files.values() is not None:
            for input_file in req.files.values():
                # logging.info(input_file.name)
                file = input_file.stream
                sampleData = np.fromfile(file, dtype=np.dtype('<u2'))
        
    if sampleData is not None:
        # Load model
        model = load_model('./AI-Model/NewAIModel.h5')
        dataset = []
        for x in sampleData:
            dataset.append(x)
        #load the sample as a numpy array and pass to model
        sampData = np.asarray(dataset)
        predictionProbability = getPredictionProbability(model, sampData)

        # Get prediction
        prediction = str(getPrediction(predictionProbability))

        # Check if rejected
        rejection = ""
        if isRejected(predictionProbability):
            rejection = ",0"
        else:
            rejection = ",1"

        prediction += rejection
        logging.info(prediction)
        return func.HttpResponse(prediction)

    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
