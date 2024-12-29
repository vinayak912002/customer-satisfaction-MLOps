import numpy as np
import pandas as pd
import mlflow
import json
from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from pipelines.utils import get_data_for_test

from config import DeploymentTriggerConfig

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data 

@step
def deployment_trigger(
    error: float,
    config : float = DeploymentTriggerConfig.max_error
):
    '''implents a simple model deployment trigger that looks at the imput model accuracy and decides if it is good enoough for deployment or not'''
    val = error<config
    print("accuracy : ", error)
    print(val)
    return val
    
@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    '''
    Get the prediction service started by the deployment pipeline,

    Args:
        pipeline_name : name of the pipeline that deployed the MLFlow prediction
        step_name : the name of the step that deployed the MLFlow prediction
        running : when this flag is set, the step only returns a running service
        model_name : the name of the model that is deployed
    '''

    # get the MLFlow deployer stack component
    mlflow_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    #fetch the existing services with the same model name, step name and model name
    existing_services = mlflow_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLFlow deployment service found for pipeline {pipeline_name}, "
            f"step {pipeline_step_name} and model {model_name} "
            f"pipeline for the {model_name} model is currently running."
        )
    
    return existing_services[0]

@step
def predictor(
    service : MLFlowDeploymentService,
    data : str
)-> np.ndarray:
    
    service.start(timeout=10)
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df= [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction
    

# if we cache the pipeline then the MLFlow experiment will not get created if the cached data is used
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path : str,
    max_error: float = 0.5,
    workers : int = 1,
    timeout : int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2_score)
    
    mlflow_model_deployer_step(
        model = model,
        deploy_decision = deployment_decision,
        workers=workers,
        timeout = timeout
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name
    )
    prediction = predictor(service=service, data=data)
    return prediction