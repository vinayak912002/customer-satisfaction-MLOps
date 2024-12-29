from pipelines.deployment_pipeline import continuous_deployment_pipeline
import click
from typing import cast

from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService

from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type = click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default = DEPLOY_AND_PREDICT,
    help = "optionally you can choose to only run the deployment"
            "pipeline to train and deploy a model (`deploy`) or to"
            "only run a prediction against the deployed model"
            "(`predict`). By default both will be run "
            "(`deploy and predict`)."
)
@click.option(
    "--max_error",
    default=0.5,
    help="maximum error permitted to deploy the model"
)
def main(config:str, max_error:float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        continuous_deployment_pipeline(
            data_path = "/home/vinayak/repos/MLOps/customer-satisfaction-mlops/data/olist_customers_dataset.csv",
            max_error = max_error,
            workers = 3,
            timeout = 60)
        
    if predict:
        inference_pipeline(
            pipeline_name = "continuous_deployment_pipeline",
            pipeline_step_name = "mlflow_model_deployer_step"
        )
        
    
    print(
        "you can run: \n"
        f"[italic green] mlflow ui --backend-store-uri {get_tracking_uri()}"
        "[/italic green]\n ...to inspect your experiment runs within MLFlow"
        "UI.\nYou can find your runs tracked within the"
        "`mlflow_example_pipeline` experiment. There you would also be able to"
        "compare two or more runs.\n\n"
    )

    # fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model"
    )

    if existing_services:
        service= cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLFlow prediction server is running locally as a daemon"
                f"process service and accepts inference requests at:"
                f"{service.prediction_url}\n"
                f"To stop service, run"
                f"[italic green] `zenml model-deployer models delete {str(service.uuid)}`[/italic green]"
            )
        elif service.is_failed:
            print(
                f"The MLFlow prediction server is in a failed state\n"
                f"Last state: '{service.status.state.value}'\n"
                f"Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLFlow deployment sever is currenntly running. The  deployment "
            "pipeline must run first to train a model and deploy it. Execute the "
            "same command with `--deploy` argument to deploy a model."
        )

if __name__ == "__main__":
    main()