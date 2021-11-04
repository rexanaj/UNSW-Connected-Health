import requests
import numpy as np
from preProcess import processData
from json import dumps, loads

# data = np.fromfile('data/0001-1/ecg.dat', dtype=np.dtype('<u2'))
# print(type(data), data.shape)
# lists = data.tolist()
# print(len(lists))
# jsonData = dumps(lists)
# myList = loads(jsonData)
# print(type(jsonData), len(jsonData), type(myList), len(myList))
# jsonData = dumps(lists)
# r = requests.post("http://localhost:7071/api/AI-Model", data=jsonData)

#test sending dat file
with open("data/0001-1/ecg.dat", "rb") as a_file:
    # print(type(a_file))
    file_dict = {"ecg.dat": a_file}
    # print(type(a_file), type(file_dict))
    response = requests.post("http://localhost:7071/api/AI-Model", files=file_dict)
    print(response.text)
