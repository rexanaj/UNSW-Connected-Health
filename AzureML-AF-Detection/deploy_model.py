# dont use this file it doenst work
from azureml.core import Workspace, Model
ws = Workspace.from_config(path=".azureml/ws_config.json")

model_name = "af-detection-Model"
end_point = "af-detection-ep"

model = Model(ws, name=model_name)

service = Model.deploy(ws, end_point, [model])
service.wait_for_deployment(show_output=True)