import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import joblib

st.write("#### Esta es una aplicacion que clasifica celulas")

df = pd.read_csv("cell_samples.csv")

atributos = ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BlandChrom', 'NormNucl', 'Mit']


columnas_x = df[atributos]

x = columnas_x.values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x)


st.image('imagenes/cell.jpg')



col1, col2, col3 = st.columns((2,1,2))

x_usuario = np.zeros(len(atributos))

with col1:

    for i in range(len(atributos)):
        x_usuario[i] = st.number_input(atributos[i],step=1)
 

with col3:
    modelo_cargado = joblib.load('modelo_arbol.joblib')
    x = x_usuario.reshape(1,-1)
    x2 = scaler.transform(x)

    y_pred = modelo_cargado.predict(x2)

    if y_pred < 0.5:
        y_pred = 0
    else:
        y_pred = 1

    st.write("Clase de la celula = ", y_pred) 

    if y_pred == 0:
        st.image("imagenes/0.jpeg")
    if y_pred == 1:
        st.image("imagenes/1.jpeg")
