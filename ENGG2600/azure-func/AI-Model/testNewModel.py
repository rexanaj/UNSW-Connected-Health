import requests
import numpy as np
from preProcess import processData
from json import dumps, loads
import sys
from csv import reader

"""
Usage: python3 testNewModel.py PLURALITY_VOTE 
Where PLURALITY_VOTE is the no. of students who agreed with the model's output
    --> Generally a value of 2 or 3
"""

"""
Get the accuracy based on the retrained model data
"""
def getAccuracy(plurality_given):
    correct = 0
    total = 0
    with open('ECG_Interpretation_New.csv') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)

        if header != None: 
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                sample_uid = row[0]
                plurality = row[5]
                model_pred = row[9]
                model_reject = row[10]

                if plurality != plurality_given:
                    continue 

                datafilePath = "./data/%s/ecg.dat" % sample_uid
                response = ''
                try: 
                    with open(datafilePath, "rb") as a_file:
                        file_dict = {"ecg.dat": a_file}
                        response = requests.post("http://localhost:7071/api/AI-Model", files=file_dict)

                    # Extract prediction and rejection
                    responses = response.text.split(',')
                    found_pred = responses[0]
                    found_reject = responses[1]

                    print(f"Sample: {sample_uid}    Model: {model_pred}, {model_reject}    Function: {found_pred}, {found_reject}")

                    if model_pred == found_pred and model_reject == found_reject:
                        correct += 1
                    
                    total += 1
                    
                except IOError:
                    print(f"{sample_uid} file not accessible")

                finally:
                    continue
    
    # Check accuracy
    if total != 0: 
        prop_correct = (correct / total) * 100
        print(f"Number correct = {correct}")
        print(f"Total number = {total}")
        print(f"Proportion correct: {prop_correct}%")

    


"""
Get the accuracy based on the student interpretation data 
"""
def getAccuracyStudentInterpretation(plurality_given):
    correct = 0
    total = 0
    with open('ECG_Interpretation_All.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)

        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                sample_uid = row[0]
                output = row[4]
                plurality = row[5]

                if plurality != plurality_given:
                    continue 

                datafilePath = "./data/%s/ecg.dat" % sample_uid
                prediction = ''
                try: 
                    with open(datafilePath, "rb") as a_file:
                        file_dict = {"ecg.dat": a_file}
                        response = requests.post("http://localhost:7071/api/AI-Model", files=file_dict)
                        prediction = response.text

                    # Format prediction
                    print(f"Sample: {sample_uid}    Model predicted: {prediction}    Student interpretation: {output}")
                        
                    if prediction == output:
                        correct += 1
                    total += 1
                    
                except IOError:
                    print(f"{sample_uid} file not accessible")

                finally:
                    continue
                    
    # Check accuracy
    if total != 0: 
        prop_correct = (correct / total) * 100
        print(f"Number correct = {correct}")
        print(f"Total number = {total}")
        print(f"Proportion correct: {prop_correct}%")


if __name__ == "__main__":
    plurality_count = 0
    if len(sys.argv) == 1:
        print("Usage: python3 testNewModel.py PLURALITY_COUNT")
        sys.exit(1)
    plurality_given = sys.argv[1]
    # getAccuracyStudentInterpretation(plurality_given)

    print(f"Plurality given: {plurality_given}")
    getAccuracy(plurality_given)
