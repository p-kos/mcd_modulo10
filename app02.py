import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# importar las librerías para el PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Agregar configuración de cabezara
st.set_page_config(page_title='Modulo 10 - MCD - PCA Componentes principales', page_icon=':shark', layout='wide')

# Título
st.title("Modulo 10 - MCD - PCA Componentes principales")

# Side bar para la navegación
# st.sidebar("Menú")

# Opciones
options = ["Carga de datos", "Análisis Eploratorio", "Análisis de Componentes Principales"]

# agregar radio
option=st.sidebar.radio("Selecciona una Opción", options=options)

@st.cache_data
def load_file(file):
    if file.name.endswith('csv'):
        df=pd.read_csv(file)
    elif file.name.endswith ('xls'):
        df=pd.read_excel(file)
    else:
        raise Exception('The file is not compatible')

if option == 'Carga de datos':
    st.subheader('Cargar Datos')
    file = st.file_uploader('Cargar archivo CSV o Excel', type=['csv', 'xlsx'])
    if file:
        df=load_file(file)
        if df is not None:
            st.session_state.df=df
            st.warning('El archivo fue correctamente cargado')
            st.write(f'El dataset tiene {df.shape.columns} columnas y {df.shape.columns} filas')
            st.write(df)
    else:
        st.warning('No se cargó el archivo')
elif option == 'Análisis Eploratorio':
    st.subheader('Análisis Exploratorio')
    if 'df' not in st.session_state:
        st.warning('Archivo no cargado')
    else:
        df = st.session_state.df
        st.subheader('Primeras 5 filas')
        st.write(df.head())
        st.subheader('Información del dataframe')
        st.write(df.describe())

        # Crear un gráfico de correlación
        st.subheader('Gráfico de correlación')
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True)
        st.pyplot(fig)


elif option == 'Análisis de Componentes Principales':
    if 'df' not in st.session_state:
        st.warning('Archivo no cargado')
    else:
        st.subheader("PCA - Componentes Principales")
        df = st.session_state.df
        # Paso 1, normalizar los datos
        st.subheader('Normalizar los datos')
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        ax1.set_title("Antes de normalizar", fontsize=15)
        ax1.scatter(df["ingreso"], df["horas_trabajadas"],
                    marker='8', s=500, c='purple', alpha=0.5)
        ax1.set_xlabel('Ingreso', fontsize=12)
        ax1.set_ylabel('Horas Trabajadas', fontsize=12)

        # Normalizar los datos
        data = pd.DataFrame(StandardScaler().fit_transform(df), columns = df.columns)

        ax2.set_title("Depsués de normalizar", fontsize=15)
        ax2.scatter(data['ingreso'], datos['horas_trabajadas'],
                    marker='8', s=500, c='red', alpha=0.5)
        ax2.set_xlabel('Ingreso', fontsize=12)
        ax2.set_ylabel('Horas Trabajadas', fontsize=12)

        st.pyplot(fig)

        # Paso 2 Matriz de Covarianza
        st.subheader('Matriz de Covarianza')
        matriz_covarianza = np.cov(data.T)

        fig, ax = plt.subplots()
        sns.heatmap(matriz_covarianza, annot=True)
        st.pyplot(fig)

        # Paso 3 Calcularo los valores y vectores propios
        st.subheader('Valores y Vectores Propios')

        valores, vectores = np.linalg.eig(matriz_covarianza)

        vector_azul = vectores[:,0]
        vector_rojo = vectores[:,1]

        st.write('EigenVector Rojo:', vector_rojo,' EigenValor Rojo', valores[1])
        st.write('EigenVector Azul:', vector_azul,' EigenValor Rojo', valores[0])

        # visualizar los vectors propios

        fig = plt.figure(figsize=(7,7))
        plt.axes().set_aspect('equal')

        plt.scatter(data['ingreso'], data['horas_trabajadas'],
                    marker='8', s=500, c='purple', alpha=0.5)
        
        plt.quiver(0,0,vector_azul[0]/abs(vector_azul[0]*valores[0]), 
                   vector_azul[1]/abs(vector_azul[1]*valores[0]),
                   angles='xy',
                   scale_units='xy',
                   scale=1,
                   color='blue')
        

        plt.quiver(0,0,vector_rojo[0]/abs(vector_rojo[0]*valores[1]), 
                   vector_rojo[1]/abs(vector_rojo[1]*valores[1]),
                   angles='xy',
                   scale_units='xy',
                   scale=1,
                   color='red')
        
        plt.xlabel('Ingreso')
        plt.ylabel('Horas Trabajadas')
        st.pyplot(fig)

        # paso 4 Proyectar los datos
        st.subheader("Proyectar los datos en los nuevos ejes")    
        datos_proyectados = pd.DataFrame(data.values @ vectores, columns=['ingreso', 'horas_trabajadas'])
        
        fig = plt.figure(figsize=(7,7))
        plt.axes().set_aspect('equal')
        plt.scatter(datos_proyectados['ingreso'], datos_proyectados['horas_trabajadas'],
                    marker='8', s=500, c='purple', alpha=0.5)
        
        # proyectar los datos
        plt.scatter(datos_proyectados['ingreso'], [-2]*len(datos_proyectados['ingreso']),
                    marker='8', s=500, c='purple', alpha=0.5)
        
        st.pyplot(fig)

        #Paso 5 Calulcar el PCA
        st.subheader('PCA con Sklearn')
        pca=PCA()
        datos=pca.fit_transform(data)

        st.write('Varianza explicada', pca.explained_variance_)
        st.write('Varianza explicada', pca.explained_variance_ratio_)
        st.write('Componentes Principales: ', pca.components_)