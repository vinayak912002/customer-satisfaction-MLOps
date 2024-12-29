from pipelines.training_pipleline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    print("Tracking uri = ",get_tracking_uri())
    #run the pipline
    train_pipeline(data_path = "/home/vinayak/repos/MLOps/customer-satisfaction-mlops/data/olist_customers_dataset.csv")