# Customer satifaction rating Prediction
This project has a ML model which predicts the customer satisfaction stores(0-5).\
This server responds to the prediction requests.The model and the prediction server both are deployed on lacal server.

Make sure you have pyhon version 3.11 on your system where you are running the project.
steps to run:-

1. install the requirements:-\
`pip install -r requirements.txt`

2. This project runs on zenml so to install zenml run:-\
`pip install "zenml[server]"`

3. the deployment pipeline uses a ZenML stack that has a MLflow experiment tracker and model deployment as components. Configuring a new stack with the two components 

```
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml stack register -a default -o default -e mlflow_tracker -d mlflow_deployer mlflow_stack
```

4. train and deploy the model:-\
`python run_deployment.py --config deploy`

5. run the prediction pipeline:-\
`python run_deployment.py --config predict`

5. start the streamlit app:-\
`streamlit run streamlit_app.py`

To start the web ui for zenml:- \
`zenml login --local`

To shut down the web ui :-\
`zenml logout --local`

## Project Structure

There are two pipelines in this project:-
1. model training 
2. model deployement

## Project flow

The steps in the training pipeline are as follows:-
1. Data ingestion
2. Cleaning the Data
3. Train model
4. evaluate model

The output for the training pipeline is the model, rmse and r2 score.

The steps in deployment model are:-
1. Take the r2 score if it fulfills the required criteria.
2. If the score fulfills the criteria then deploy the service.

There is one more pipline which sets up the inference server.
The inference pipeline:-

1. Takes the deployment service and the data for testing.
2. The output is an ndarray which includes all predictions. 


The implementation of each step is defined in the steps folder.

to open the mlflow web ui run :-\
`mlflow ui --backend-store-uri <TRACKING_URI>`

to get the UUID run:-\
`zenml model-deployer models list`

to start an existing model service:-\
`zenml model-deployer models start <UUID>`

to delete existing model service:-\
`zenml model-deployer models delete <UUID>`