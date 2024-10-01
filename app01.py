# Activate Env:  source .venv/bin/activate
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

st.title("Modulo 10 - Maestría en Ciencia de Datos")

st.subheader('Cargar Datos')

uploaded_file = st.file_uploader("Sube el archivo", type="csv")

df = pd.read_csv(uploaded_file)

st.write(df)

st.subheader('Información del Dataset')
st.write(df.info())


st.write(df.isnull().sum())

# Eliminar nulos
st.subheader("Eliminar valores nulos")
df.dropna(inplace=True)
st.write(df.isnull().sum())

# gráfico de barras
st.subheader('Gráfico de barras')
fig = px.histogram(df, x="sexo")
st.plotly_chart(fig)

# Agregar un checkbox para seleccionar columnas y graficar

st.subheader('Gráfico de barras interactivo')
columns = df.columns.tolist()
selected_columns = st.multiselect('Selecciona las columnas', columns)

# crear un gráfico de correlacion con las columnas seleccionadas

import seaborn as sns

st.write(selected_columns)

if len(selected_columns) > 0:
    st.write('Gráfico de correlación')
    fig,ax = plt.subplots()
    ax.scatter(df[selected_columns[0]], df[selected_columns[1]], color="blue", alpha=0.5)
    st.pyplot(fig)
else:
    st.write('Selecciona una columna')


# agregar la librería para dividir dataset en entrenamiento y test

from sklearn.model_selection import train_test_split

X = df.drop(columns=['sexo'])
y = df['sexo']

# agregar un slider para seleccionar el tamaño del test

test_size = st.slider('Tamaño del Test', min_value=0.01, max_value=0.99, value=0.3, step = 0.1 )

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = test_size)

st.write('tamaño del entrenamiento ', X_train.shape)
st.write('tamaño del test', y_test.shape)