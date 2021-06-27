import logging
import sys
import azure.functions as func
from .loadModel import getPrediction
from tensorflow.keras.models import load_model
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    sampleNum = req.params.get('sample')
    # logging.info(sampleNum)
    if not sampleNum:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            sampleNum = req_body.get('sample')

    if sampleNum:
        # Load model
        model = load_model('./AI-Model/ECGClassificationModel.h5')

        # Check for valid sampleNum
        datafilePath = f"./AI-Model/data/{sampleNum}/ecg.dat"
        if not os.path.exists(datafilePath):
            return func.HttpResponse(f"The sample number {sampleNum} is invalid.")

        datafile = open(datafilePath, 'rb')
        prediction = getPrediction(model, sampleNum, datafile)
        logging.info(prediction)

        return func.HttpResponse(f"The prediction for Sample {sampleNum} is '{prediction}'.")

    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
