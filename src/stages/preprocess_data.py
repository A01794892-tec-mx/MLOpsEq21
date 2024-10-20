import argparse
from typing import Text
import yaml
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def data_pre_proc(config_path: Text) -> None:
    # Read the YAML configuration file
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    data = pd.read_csv(config['data_preproc']['in'] ,sep=',', header='infer')
    data.rename(columns={' RH':'RH', ' Ws': 'Ws', 'Classes  ':'Classes', 'Rain ':'Rain'}, inplace=True)
    data['Classes'] = data['Classes'].str.strip()
    data_limpia = data.dropna()

    columnas_categoricas = ['Classes', 'Region']
    columnas_enteras = ['day', 'month', 'year']
    columnas_continuas = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

    # Convert columns to appropriate types
    data_transformada = data_limpia.copy()
    data_transformada[columnas_categoricas] = data_limpia[columnas_categoricas].astype('category')
    data_transformada[columnas_enteras] = data_limpia[columnas_enteras].astype('int64')
    data_transformada[columnas_continuas] = data_limpia[columnas_continuas].astype('float64')

    y = data_transformada['Classes']
    data_transformada.drop(columns=['Classes'], inplace=True)
    X = data_transformada.select_dtypes(include=['float64', 'int64', 'category'])

    X = X.drop(columns=['day', 'month', 'year'])

    # Split the data into training and test sets (80% training, 20% test)
    X_train_base, X_test_base, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline for numeric continuous variables: scaling + PCA for dimensionality reduction
    numericas_pipeline = Pipeline( steps=[
        ('minmax', MinMaxScaler()),
        ('scaler', StandardScaler()),
        ('PCA', PCA(n_components=0.95))
    ] )

    # Pipeline for categorical variables: one-hot encoding
    catOHE_pipeline = Pipeline( steps=[
        ('OneHotEncoder', OneHotEncoder())
    ] )

    # Define which columns go through which pipeline
    columnas_categoricas = ['Region']
    ct = columnasTransformer = ColumnTransformer(transformers=[
            ('numericas_continuas', numericas_pipeline, columnas_continuas),
            ('categoricas', catOHE_pipeline, columnas_categoricas)
            ])

    X_train = ct.fit_transform(X_train_base)
    X_test  = ct.transform(X_test_base)

    pd.DataFrame(X_train).to_csv(config['data_preproc']['out_X_train'], index=False)
    pd.DataFrame(X_test).to_csv(config['data_preproc']['out_X_test'], index=False)
    pd.DataFrame(y_train).to_csv(config['data_preproc']['out_y_train'], index=False)
    pd.DataFrame(y_test).to_csv(config['data_preproc']['out_y_test'], index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_pre_proc(config_path=args.config)