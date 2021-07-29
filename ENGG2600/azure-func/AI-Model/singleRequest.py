import sys
import requests
import numpy as np
from preProcess import processData
from json import dumps, loads

"""
This program sends a request to the Azure function for one .dat file

Steps for running: 
1) Ensure current directory is the `azure-func` directory
2) Run `func start` in one terminal
3) In a separate terminal, run change to the AI-Model directory (`cd AI-Model`)
4) Run this program using the following usage pattern

Usage:   python3 singleRequest.py PATH_TO_DAT_FILE
Example: python3 singleRequest.py data/0001-1/ecg.dat
"""

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python3 singleRequest.py PATH_TO_DAT_FILE")
        sys.exit(1)
    
    filepath = sys.argv[1]
    with open(filepath, "rb") as a_file:
        file_dict = {"ecg.dat": a_file}
        response = requests.post("http://localhost:7071/api/AI-Model", files=file_dict)
        print(f"Azure function returned: {response.text}")
        prediction = response.text.split(',')[0]
        rejection = response.text.split(',')[1]

        # Print prediction
        if prediction == "1":
            print(f"Prediction:\t{prediction}\tAFIB")
        elif prediction == "2":
            print(f"Prediction:\t{prediction}\tNSR")
        elif prediction == "3":
            print(f"Prediction:\t{prediction}\tother ")
        elif prediction == "4":
            print(f"Prediction:\t{prediction}\tToo noisy ")
        else:
            print(f"Prediction:\t{prediction}\tModel failed")

        # Print rejection
        if rejection == "0":
            print(f"Rejection:\t{rejection}\tSample accepted")
        else:
            print(f"Rejection:\t{rejection}\tSample rejected")
