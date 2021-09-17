from azureml.core import Workspace, Model
import tensorflow as tf
ws = Workspace.from_config(path=".azureml/ws_config.json")
model = Model.register(workspace=ws, 
                    model_path="./ModelTrainedOnCambellSamples.h5", 
                    model_name="af-detection-Model",
                    model_framework=Model.Framework.TENSORFLOW,
                    model_framework_version=tf.__version__ )