import argparse
from typing import Text
import yaml
import pandas as pd

def data_cleanup(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    data = pd.read_csv(config['data_cleanup']['dataset_algerian_forest_fires'] ,sep=',', header='infer')

    print(data)

    data.rename(columns={' RH':'RH', ' Ws': 'Ws', 'Classes  ':'Classes', 'Rain ':'Rain'}, inplace=True)

    #data.info()

    #print(f"Classes: {data['Classes'].unique()} \n")

    #print(f"Region {data['Region'].unique()} \n")

    #Detectamos que los valores de Classes contienen espacios extras por lo cual procedemos a eliminarlos
    #El valor nulo se va a limpiar en el siguiente paso
    data['Classes'] = data['Classes'].str.strip()

    #Imprimimos de nuevo los valores unicos:
    #print(f"Classes: {data['Classes'].unique()} \n")

    valores_nulos = data.isnull().sum()
    data_limpia = data.dropna()

    #Realizamos la transformacion de variables en el dataframe a su correcto formato
    columnas_categoricas = ['Classes', 'Region']
    columnas_enteras = ['day', 'month', 'year']
    columnas_continuas = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

    data_transformada = data_limpia.copy()
    data_transformada[columnas_categoricas] = data_limpia[columnas_categoricas].astype('category')
    data_transformada[columnas_enteras] = data_limpia[columnas_enteras].astype('int64')
    data_transformada[columnas_continuas] = data_limpia[columnas_continuas].astype('float64')

    data_transformada.to_csv(config['data_cleanup']['dataset_algerian_forest_fires_clean'], index=False)

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_cleanup(config_path=args.config)