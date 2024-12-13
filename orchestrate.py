import logging
import numpy as np
import pandas as pd
import skops.io as sio
from typing import Tuple
from prefect import flow, task
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PowerTransformer



@task(name="Load data", description="Task to load data from a data directory")
def load_data(filename: str) -> pd.DataFrame:
    data = pd.read_csv(filename)
    return data



@task(name="Split data", description="Split data into Training, Validation , and Test set")
def split_data(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    
    df = load_data(filename)
 
    X = df.drop(columns=["RiskLevel"])
    y = df["RiskLevel"]
   
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, 
                                                                   test_size=0.15, 
                                                                   random_state=5)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, 
                                                          test_size=0.1, 
                                                          random_state=5)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test



@task(name="Process data", description="Handle skweness, encode the label, apply resampling, and scale the data")
def data_preprocessing(X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame, 
                  y_train: pd.Series, y_valid: pd.Series, y_test: pd.Series,
                  transform_columns: list = None 
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    if transform_columns is None:
        transform_columns = X_train.columns
    
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    X_train[transform_columns] = power_transformer.fit_transform(X_train[transform_columns])
    X_valid[transform_columns] = power_transformer.transform(X_valid[transform_columns])
    X_test[transform_columns] = power_transformer.transform(X_test[transform_columns])

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid) 
    y_test_encoded = label_encoder.transform(y_test) 
    
    smote = SMOTE(random_state=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_encoded)
     
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train_resampled, y_valid_encoded, y_test_encoded



@task(name="Train model", description="Train the data and make prediction on the test set")
def training_predict(X_train_scaled: np.ndarray, X_test_scaled:np.ndarray, y_train_encoded: np.ndarray) -> np.ndarray:
    best_params = {
        "n_neighbors": 31,
        "weights": "distance",
        "algorithm": "ball_tree",
        "leaf_size": 20,
        "p": 1,
        "metric": "euclidean"
    }

    knn_model = KNeighborsClassifier(**best_params)
    knn_model.fit(X_train_scaled, y_train_encoded)
    y_pred = knn_model.predict(X_test_scaled)
    
    return knn_model, y_pred



@task(name="Evaluate model", description="Evaluate the performance of the model on the test set")
def evaluate_model(y_test_encoded: np.ndarray, prediction: np.ndarray) -> None:
    accuracy = accuracy_score(y_test_encoded, prediction)
    f1 = f1_score(y_test_encoded, prediction, average='weighted')
    recall = recall_score(y_test_encoded, prediction, average='weighted')
    
    print(f"Accuracy : {accuracy*100:.2f}%\t F1 : {f1*100:.2f}%\t Recall : {recall*100:.2f}%")



@task(name="Save model", description="Save the model for further use")
def save_model(knn_model: KNeighborsClassifier):
    sio.dump(knn_model, "mhr_knn_model.skops")
    
    
    
logger = logging.getLogger("prefect")

@flow(log_prints=True, name="workflow")
def run_ml_workflow(filename: str="data/Maternal-Health-Risk-Data-Set.csv"):
    logger.info("Flow started.")
    logger.info(f"Using dataset: {filename}")
    
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(filename)

    X_train_scaled, X_valid_scaled, X_test_scaled, y_train_encoded, y_valid_encoded, y_test_encoded = data_preprocessing(
        X_train, X_valid, X_test, 
        y_train, y_valid, y_test,
        transform_columns=["HeartRate", "BodyTemp", "BS"]
        )
    
    knn_model, predictions = training_predict(X_train_scaled, X_test_scaled, y_train_encoded)
    evaluate_model(y_test_encoded, predictions)
    save_model(knn_model)
    


if __name__ == "__main__":
    run_ml_workflow()