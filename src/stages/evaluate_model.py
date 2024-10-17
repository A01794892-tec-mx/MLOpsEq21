import pickle

import argparse
from typing import Text
from sklearn import metrics
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

def evaluate_model(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    # Cargar el modelo desde el archivo .pkl
    with open(f"{config['train_lr']['out']}/{config['train_lr']['model_LR']['modelName']}_{config['dvc_version']}.pkl", 'rb') as f:
        modelLR = pickle.load(f)

    X_train =   pd.read_csv(config['evaluate_model']['in_X_train'])
    X_test  =   pd.read_csv(config['evaluate_model']['in_X_test'])
    y_train =   pd.read_csv(config['evaluate_model']['in_y_train'])
    y_test  =   pd.read_csv(config['evaluate_model']['in_y_test'])

    print(">>Exactitud (Accuracy) de los conjuntos de Entrenamiento y Validación con Logistic Regresion:")
    y_pred_trainRL = modelLR.predict(X_train)
    y_pred_testRL = modelLR.predict(X_test)
    print('accuracy-train', metrics.accuracy_score(y_train, y_pred_trainRL))
    print('accuracy-test', metrics.accuracy_score(y_test, y_pred_testRL))

    print("\n>>Matriz de Confusión:")
    print(metrics.confusion_matrix(y_test, y_pred_testRL))

    print("\n>>Reporte varias métricas:")
    print(metrics.classification_report(y_test, y_pred_testRL))

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)