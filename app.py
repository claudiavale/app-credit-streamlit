import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
#from sklearn import StandarScaler
from sklearn.preprocessing import StandardScaler
#from sklearn.externals import joblib
import joblib

st.write("### Esta es una aplicación que clasifica la aprobación de créditos")

df = pd.read_csv('credit-approval.csv')

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

#convertimos valores no numericos en numericos 
for col in df:
    if df[col].dtypes=='object':
        df[col]=LE.fit_transform(df[col])

features = ['IncumplimientosPrevios', 'TiempoEmpleado', 'PuntajeCredito', 'Ingresos', 'Deuda', 'Edad']

columnas_x = df[features]

x = columnas_x.values

scaler = StandardScaler().fit(x)



image_credito = Image.open('Imagenes/creditcard.jpg')
st.image(image_credito)



col1, col2, col3 = st.columns((2,1,2))

x_usuario = np.zeros(len(features))

with col1:
    for i in range(len(features)):
        x_usuario[i] = st.number_input(features[i], step=1)
        
        
with col3:
    red_neural = joblib.load('modelo_arbol.joblib')
    x = x_usuario.reshape(1,-1)
    x2 = scaler.transform(x)
    
    y_pred = red_neural.predict(x2)
    
    if y_pred < 0.5:
        y_pred = 0
    else:
        y_pred = 1
        
    st.write('Aprobación de tarjeta de crédito = ', y_pred)

    
    if y_pred == 0:
        im = Image.open("imagenes/denegado.jpg")
        st.image(im)
    if y_pred == 1:
        im = Image.open("imagenes/aprobado.jpg")
        st.image(im)