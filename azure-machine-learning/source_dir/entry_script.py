from azureml.core import Workspace
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import WebService
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.model import Model
from azureml.core.conda_dependencies import CondaDependencies

import json
def init():
    print("This is Init")    

def run(data):
    test = json.loads(data)
    print(f"recieved data{test}")
    return f"test is test"