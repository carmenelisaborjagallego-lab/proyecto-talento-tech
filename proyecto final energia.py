# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 16:20:10 2025

@author: USUARIO
"""

# ==========
# 1. IMPORTAR LIBRERÍAS
# ==========

# Pandas: manejo de tablas de datos (DataFrame)
import pandas as pd

# NumPy: operaciones numéricas (lo usaremos para algunas funciones)
import numpy as np

# Matplotlib y Seaborn: visualización básica (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Importas la clase OneHotEncoder, que sirve para convertir texto en números mediante One Hot Encoding.
from sklearn.preprocessing import OneHotEncoder

# StandardScaler: escalamiento de variables numéricas
from sklearn.preprocessing import StandardScaler

# train_test_split: separar datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Opcional: un modelo simple para demostrar uso (ej: Regresión Logística)
from sklearn.linear_model import LogisticRegression

import category_encoders as ce
import category_encoders as ce

# ===== IMPORTES ADICIONALES PARA MODELOS Y PIPELINES =====

# Modelos supervisados
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Modelo no supervisado
from sklearn.cluster import KMeans

# Métricas
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Herramientas avanzadas
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Guardar pipeline entrenado
#PIPELINES permite encadenar todos los pasos de tu proceso de Machine Learning en un solo flujo ordenado, desde el preprocesamiento hasta el modelo final.

import joblib


#CARGA DE DATASET
df1_xlsx=pd.read_excel("Datos Zaragoza1.xlsx")
print(df1_xlsx)

#copy
df_csv=df1_xlsx.copy()
print(df_csv)

#borrar columnas
df_csv.drop(columns=['Coordenadas_gps','Municipio',"Estado_edificio","Dias_hasta_expiracion"], inplace=True)
print(df_csv.columns)

#ORDENAR DE menor a mayor
# Si la columna no esta en formato datetime, primero conviértela
df_csv['Fecha_emision'] = pd.to_datetime(df_csv['Fecha_emision'])
df2 = df_csv.sort_values(by='Fecha_emision')


#filtro para borar filas mayores a 2016
print(df2[df2["Anio_emision"]>2016])

# Eliminar las filas en las posiciones 
df3 = df2.drop(df2.index[57111:179852])


#eliminacion dato atipico
df3=df3.drop(df3.index[0])

#eliminacion de duplicados dejando solo la primera aparicion
print(df3.drop_duplicates())
df3 = df3.drop_duplicates()
print(df3.info())

print(df3[df3["Anio_emision"]>2017])
######empty Dataframe###################

#filtro para borrar filas con guion y con cero consumo 
print(df3.columns)
df4 = df3[df3['Clasificacion_consumo'] != '-']
df4 = df4[df4['ConsumoKWh/m2/Anio'] != 0]

##### CONVERTIR CATEGORIAS DE CONSUMO Y DE EMISIONES A TIPO BINARIO (0 Y 1) ###############
## Clasificacion_emisiones
####  A, B, C, D Y E SON 0  ####  F Y G SON 1
print(df4.columns)
df4['Clasificacion_Emisiones'] = df4['Clasificacion_Emisiones'].replace({'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,'F': 1, 'G': 1})
print(df4.info())
## Clasificacion_consumo
####  A, B, C, D Y E SON 0  ####  F Y G SON 1
print(df4.columns)
df4['Clasificacion_consumo'] = df4['Clasificacion_consumo'].replace({'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,'F': 1, 'G': 1})
print(df4.info())

###### CONVERTIR TIPO DE EDIFICIO Y PROVINCIA A NUMEROS ###########
print(df4['Tipo_edificio'].value_counts())
#Tipo_edificio
#Vivienda individual 0
#Bloque completo 1
#Local 2
#Unifamiliar 3
#Edificio completo 4
df4['Tipo_edificio'] = pd.factorize(df4['Tipo_edificio'])[0]

print(df4['Provincia'].value_counts())
#Provincia
#ZARAGOZA 0
#HUESCA 1
#TERUEL 2
df4['Provincia'] = pd.factorize(df4['Provincia'])[0]
print(df4.info())


###### EXPORTAR DATA SET LIMPIO ###########

df4.to_csv("proyecto_limpio.csv", index=False)

df = pd.read_csv("proyecto_limpio_ml.csv")
print(df.info())

