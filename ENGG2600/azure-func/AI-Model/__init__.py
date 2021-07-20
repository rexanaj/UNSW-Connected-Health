from json.decoder import JSONDecodeError
import logging
import sys
import azure.functions as func
import numpy as np
from json import loads, dumps
from .loadModel import getPrediction
from tensorflow.keras.models import load_model
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    '''
    if jsonData:
        logging.info("Sample data has been found")
        string1 = str(type(jsonData))
        string2 = str(len(jsonData))
        array = np.array(jsonData)
        string3 = str(type(array))
        logging.info(string1 + string2 + string3)
        return func.HttpResponse("data has been found and return the otucome")
    sampleNum = req.params.get('sample')
    # logging.info(sampleNum)
    if not sampleNum:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            sampleNum = req_body.get('sample')
    '''
    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else: 
        #this is not complete the function needs to check for the sample file or list
        sampleData = req_body
    if sampleData:
        # Load model
        model = load_model('./AI-Model/ECGClassificationModel.h5')
        #load the sample as a numpy array and pass to model
        dataset = np.asarray(sampleData)
        prediction = getPrediction(model, dataset)
        logging.info(prediction)

        return func.HttpResponse(f"The prediction for Sample is '{prediction}'.")

    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
