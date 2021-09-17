#script used to create the machine learning workspace using python
#need to add subscription ID which can be found in settings on azure portal
#or alternatively on details of other resources/services used
from azureml.core import Workspace, Model
ws = Workspace.create(name='AF-detection-MLService',
                      subscription_id='8d37b24a-d98d-42dd-bda4-abf4a68c5d43', #change this, I have left it for reference
                      resource_group='<name for resource group>',
                      create_resource_group=True,
                      location='australiaeast'
                     )

ws.write_config(path="./.azureml", file_name="ws_config.json")

model = Model.register(workspace=ws, model_path="./ModelTrainedOnCambellSamples.h5", model_name="af-detection-Model")