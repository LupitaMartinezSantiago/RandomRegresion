from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.notebook_runner import ejecutar_notebook  # Importar la función para ejecutar el notebook

app = Flask(__name__)

# Ruta del archivo de dataset específico
DATASET_PATH = '/home/lupita/Documentos/ForestRegresion/TotalFeatures-ISCXFlowMeter.csv'

# Ruta del notebook
NOTEBOOK_PATH = '/home/lupita/Documentos/ForestRegresion/Random Forest.ipynb'


@app.route('/')
def index():
    # Ejecutar el notebook y obtener los resultados
    notebook_output = ejecutar_notebook(NOTEBOOK_PATH)

    # Código original para procesar el dataset y generar la gráfica
    df = pd.read_csv(DATASET_PATH)
    
    # Identificar y convertir columnas categóricas
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # Separar características (X) y etiquetas (y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Escalado de características
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Separar en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Definir el número de datos de entrenamiento que deseas utilizar
    num_training_samples = 5000  
    X_train = X_train[:num_training_samples]
    y_train = y_train[:num_training_samples]

    # Entrenamiento del modelo
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Graficar resultados
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Resultados de Predicción')
    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)

    # Renderizar la plantilla con el contenido del notebook y la gráfica
    return render_template('index.html', plot_url=plot_path, notebook_output=notebook_output)


if __name__ == '__main__':
    app.run(debug=True)
