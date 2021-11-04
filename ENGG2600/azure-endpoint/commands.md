## Azure CLI commands 

### Creating the endpoint
az ml endpoint create -n my-endpoint -f endpoint.yaml

### Invoking the endpoint 
az ml invoke -n my-endpoint --request-file ecg_sample.json

### Deleting the endpoint
az ml endpoint delete -n my-endpoint