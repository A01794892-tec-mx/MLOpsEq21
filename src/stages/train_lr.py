import argparse
from typing import Text
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

def train_lr(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    X_train=pd.read_csv(config['train_lr']['in_X_train'])
    y_train=pd.read_csv(config['train_lr']['in_y_train'])

    modeloRL = LogisticRegression(penalty=config['train_lr']['model_LR']['penalty'],
                                  C=config['train_lr']['model_LR']['C'],
                                  solver=config['train_lr']['model_LR']['solver'],
                                  max_iter=config['train_lr']['model_LR']['max_iter'],
                                  random_state=config['train_lr']['model_LR']['random_state'])

    modeloRL.fit(X_train,np.ravel(y_train))

    with open(config['train_lr']['model_LR']['out'], 'wb') as f:
        pickle.dump(modeloRL, f)

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_lr(config_path=args.config)