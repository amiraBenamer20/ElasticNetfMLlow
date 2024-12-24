import os
import sys

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

os.environ["MLFLOW_TRACKING_URI"]="http://ec2-15-236-133-44.eu-west-3.compute.amazonaws.com:5000/"
def eval_metrics(actual,pred):
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae  = mean_absolute_error(actual,pred)
    r2_sc = r2_score(actual,pred)
    return rmse, mae, r2_sc


if __name__=="__main__":

    ## Data ingestion-Reading the dataset --wine quality dataset
    try:
        data = pd.read_csv("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv", sep=";")
    except Exception as e:
        logger.exception("Unable to download the data")


    ## Split the data (train, test)
    train,test = train_test_split(data, test_size=0.2)
    train_X = train.drop(["quality"], axis=1)
    train_y = train[["quality"]]

    test_X = test.drop(["quality"], axis=1)
    test_y = test[["quality"]]

    ## ElasticNet paramters
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_X, train_y)

        predicted_qualities = lr.predict(test_X)
        (rmse, mae, R2_sc) = eval_metrics( test_y, predicted_qualities)

        print(f"rmse: {rmse}")
        print(f"mae: {mae}")
        print(f"r2_sc: {R2_sc}")

        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)
        
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2_score",R2_sc)
        
        

        ## for the remote server AWS
        remote_server_uri="http://ec2-15-236-133-44.eu-west-3.compute.amazonaws.com:5000/"
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!="file":
            mlflow.sklearn.log_model(lr,"model",registered_model_name="ElasticNetWineModel")
        else:
            mlflow.sklearn.log_model(lr,"model")        



