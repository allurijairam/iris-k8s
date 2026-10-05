import sklearn
import argparse
import mlflow
from mlflow.models import infer_signature

from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


model_name = "iris_model"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators",type=int,default=1,help="Number of estimators for the random forest model")
    parser.add_argument("--max-depth",type=int,default=5,help="Maximum depth of the random forest model")
    return parser.parse_args()

def train(args,model_name):
    n_estimators = args.n_estimators
    max_depth = args.max_depth

    iris_data = load_iris() 


    X_train,X_test,y_train,y_test = train_test_split(iris_data['data'],iris_data['target'],test_size=0.2,random_state=42)


    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_experiment")

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        model.fit(X_train,y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test,preds)
        f1 = f1_score(y_test,preds,average='weighted')
        precision = precision_score(y_test,preds,average='weighted')
        recall = recall_score(y_test,preds,average='weighted')

        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        mlflow.log_metrics({"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall})


        info = mlflow.sklearn.log_model(sk_model=model,
        name="iter1",
        signature=infer_signature(X_train,model.predict(X_train)),
        registered_model_name=model_name,
        skops_trusted_types=["sklearn.tree._tree.Tree"])
        
        version = info.registered_model_version

        print (f"model registered with run_id:{run.info.run_id} and version: {version}")


def main():
    args = parse_args()
    train(args,model_name)
    

if __name__ == "__main__":
    main()





