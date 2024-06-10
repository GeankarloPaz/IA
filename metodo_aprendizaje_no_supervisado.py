# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def generar_datos():
    # Leer el archivo CSV
    df = pd.read_csv('datos_simulados.csv')

    return df

def entrenar_modelo(df):
    # Definir las variables dependientes e independientes
    X = df.drop('tiempo_viaje', axis=1)
    y = df['tiempo_viaje']

    # Codificar las variables categóricas
    X = pd.get_dummies(X)

    # Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar un modelo de bosque aleatorio
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Aquí está la corrección
    model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular el error cuadrático medio
    mse = mean_squared_error(y_test, y_pred)
    print(f'Error cuadrático medio: {mse}')

def main():
    df = generar_datos()
    print(df)  # Imprimir todo el DataFrame
    entrenar_modelo(df)

if __name__ == "__main__":
    main()
