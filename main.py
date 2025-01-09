import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Función para entrenar el modelo y hacer predicciones
def entrenar_modelo(X_train, X_test, y_train, y_test, kernel, C, epsilon):
    modelo_svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
    modelo_svr.fit(X_train, y_train)
    y_pred = modelo_svr.predict(X_test)
    return y_pred, modelo_svr

# Interfaz con Streamlit
st.title("Predicción del Clima con SVM")
st.write("Ajusta los parámetros para entrenar un modelo SVM y visualizar los resultados.")

# Subir dataset
dataset_file = st.file_uploader("Sube tu dataset en formato CSV", type="csv")
if dataset_file is not None:
    df = pd.read_csv(dataset_file)
    st.write("Vista previa del dataset:")
    st.write(df.head())

    # Selección de la variable objetivo
    target = st.selectbox("Selecciona la variable objetivo (target):", df.columns)
    features = st.multiselect("Selecciona las características (features):", [col for col in df.columns if col != target])

    if target and features:
        # Preprocesamiento
        X = df[features]
        y = df[target]

        # Escalado
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Selección de kernel y parámetros
        kernel = st.selectbox("Selecciona el kernel:", ["linear", "poly", "rbf", "sigmoid"])
        C = st.slider("C (Regularización):", 0.1, 10.0, 1.0)
        epsilon = st.slider("Epsilon:", 0.01, 1.0, 0.1)

        # Entrenar el modelo
        if st.button("Entrenar Modelo"):
            y_pred, modelo = entrenar_modelo(X_train, X_test, y_train, y_test, kernel, C, epsilon)

            # Métricas
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"**MAE:** {mae:.2f}")
            st.write(f"**MSE:** {mse:.2f}")

            # Visualización
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            ax.set_xlabel("Valores Reales")
            ax.set_ylabel("Predicciones")
            ax.set_title("Predicciones vs Valores Reales")
            st.pyplot(fig)
