from azureml.core.webservice import LocalWebservice
from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core import Workspace
from azureml.core.model import Model

ws = Workspace.from_config(path=".azureml/ws_config.json")
model = Model.register(workspace=ws, 
                    model_path="./ModelTrainedOnCambellSamples.h5", 
                    model_name="af-detection-Model",
                    model_framework=Model.Framework.TENSORFLOW,
                    model_framework_version=tf.__version__ )
env = Environment(name="af-environment")
inference_config = InferenceConfig(environment=env,
                    source_directory="./source_dir",
                    entry_script="./entry_script.py")
deployment_config = LocalWebservice.deploy_configuration(port=6789)
service = Model.deploy(ws, 
            "myservice", 
            [model],
            inference_config,
            deployment_config,
            overwrite=True)
service.wait_for_deployment(show_output=True)
print(service.get_logs())