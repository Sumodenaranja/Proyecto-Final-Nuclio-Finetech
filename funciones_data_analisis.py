#!/usr/bin/env python
# coding: utf-8

# # Funciones Data Analisis

# ## Importacion de librerias

# In[1]:


# Librerías estándar
import os
import sys
import warnings

# Manipulación de datos
import pandas as pd
import numpy as np

# Configuración de warnings
warnings.filterwarnings('ignore')

# Análisis de nulos
import missingno as msno

# Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Textos
import unicodedata
from fuzzywuzzy import process
import re


# ## Funciones de lectura de datos

# ### Obtener el nombre de los archivos de una carpeta

# In[2]:


def carga_archivos(path):
    """
    Obten los nombres de los archivos .csv, .xlsx y .xls que esten en una carpeta.

    Args:
        path (str): Ruta al directorio con los archivos.

    Returns:
        list[str]: Lista con los nombres de archivos .csv, .xlsx y .xls.
    """

    # Lista de los archivos en el directorio.
    lista = os.listdir(path)

    # Filtrado archivos .csv y .xlsx.
    files = [file for file in lista if file.endswith(('.csv', '.xlsx', '.xls'))]

    return files


# ### Funcion para leer archivos CSV o Excel.

# In[3]:


def leer_archivo(ruta_completa):
    """
    Obten la ruta completa de un archivo en formato CSV o Excel y lo transforma a un DataFrame

    Args:
        ruta_completa (str): Ruta completa al archivo CSV o Excel.

    Returns:
        pandas.DataFrame: DataFrame con los contenidos de el CSV o Excel.
    """

    try:
        ruta_completa=ruta_completa.strip()
        _, extension = os.path.splitext(ruta_completa.lower())

        if extension == '.csv':
            df = pd.read_csv(ruta_completa, sep=None, engine='python') # Forzamos que python interprete el separador del .csv
        elif extension in ('.xlsx', '.xls'):
            df = pd.read_excel(ruta_completa)
        else:
            print("Error: Formato no compatible")
            return None

        return df

    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en la ruta '{ruta_completa}'.")
        return None

    except Exception as e:
        print(f"Error inesperado: {e}")
        return None


# ### Funcion para leer varios archivos

# In[ ]:


def leer_archivos(files,path):
    """
    Obten un dicionario en el qual la llave es el nombre de un archivo CSV o Excel, 
    y el valor es el DataFrame correspondiente.

    Args:
        archivos (List[str]): Lista con los nombres de los archivos a cargar.
        path (str): Ruta a la carpeta que contiene archivos .csv, .xlsx y .xls.

    Returns:
        dict{str: pandas.DataFrame}: Diccionario que mapea cada nombre de archivo
                                     a su DataFrame correspondiente.
    """
    dict_data={}

    for file in files:
        ruta = os.path.join(path, file)
        df_temp = leer_archivo(ruta)
        dict_data[file] = df_temp
        print(f"{file}: {df_temp.shape}") 

    return dict_data


# ## Funciones de Exploracion de Datos

# In[5]:


def exploracion_datos(df):
    """
    Recibe un DataFrame, y muestra una exploracion inicial de los datos

    Args:
        df (pandas.DataFrame): DataFrame con los datos que deseamos visualizar

    """
    print('Exploración inicial de datos:')
    print('*'*100)

    # Información general del dataframe.
    num_filas, num_columnas = df.shape
    print(f'El numero de filas es: {num_filas}\nEl numero de columnas es: {num_columnas}')
    print('*'*100)

    # Exploracion visulal de las primeras, últimas y aleatorias filas del dataframe.
    print('Las 5 primeras filas del dataframe son:')
    display(df.head())
    print('*'*100)
    print('Las 5 últimas filas del dataframe son:')
    display(df.tail())
    print('*'*100)
    print('Muestra aleatoria de 5 filas del dataframe:')
    display(df.sample(5))
    print('*'*100)

    # Estadisticos descriptivos del dataframe.
    print('Estadísticos descriptivos del dataframe:')
    display(df.describe())
    print('*'*100)

    # Resumen de tipologia de datos, visualizacion de nulos y valores unicos.
    print('Resumen de tipología de datos, visualización de nulos y valores únicos:')
    df_tipos=df.dtypes.to_frame(name='Tipos de datos')
    df_nulos=df.isnull().sum().to_frame(name='Nulos')
    df_porc_nulos = (df.isnull().sum() / len(df) * 100).to_frame(name='Porcentaje Nulos')
    df_valores_unicos = pd.DataFrame(df.apply(lambda x: x.unique()))
    df_valores_nunicos = pd.DataFrame(df.apply(lambda x: x.nunique()))
    df_por_valores_nunicos=pd.DataFrame(df.apply(lambda x: x.nunique())/df.shape[0]*100)
    df_valores_unicos.rename(columns={0:'Valores unicos'}, inplace=True)
    df_valores_nunicos.rename(columns={0:'Numero valores unicos'}, inplace=True)
    df_por_valores_nunicos.rename(columns={0:'Porcentaje valores unicos'}, inplace=True)
    df_exploracion = pd.concat([df_tipos, df_nulos, df_porc_nulos,df_valores_nunicos,df_por_valores_nunicos,df_valores_unicos], axis=1)

    # MOSTRAR el resumen final
    display(df_exploracion)
    print('*'*100)

    return df_exploracion


# ## Funciones de limpieza de datos

# ### Funcion para normalizar textos

# In[6]:


def normalizar_textos(df):
    """
    Normaliza los datos de tipo texto de un DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame con datos de tipo texto sin normalizar.

    Returns:
        pandas.DataFrame: DataFrame con los datos de tipo texto normalizados.
    """

    for col in df.select_dtypes(include=['object']).columns: # Iteramos sobre columnas de tipo object.
        df[col]=df[col].str.lower() # Convertimos a minúsculas.
        df[col]=df[col].str.strip() # Eliminamos espacios en blanco al inicio y final.
        df[col]=df[col].str.replace(r'\.+$', '', regex=True) # Eliminamos los '.' al final de las cadenas de texto.
        df[col]=df[col].str.replace(r'(?<=\w)\.+(?=\w)', '_', regex=True) # Reemplazamos los '.' entre palabras por '_'.
        df[col]=df[col].str.replace(r'(?<=\w)-+(?=\w)', '_', regex=True) # Reemplazamos los '-' entre palabras por '_'.

    return df


# ### Funcion para normalizar binarios

# In[ ]:


def normalizar_binario(df,columna_a_normalizar):
    """
    Transforma los datos de una columna que contiene 'yes' y 'no' a datos binarios (1 y 0)

    Args:
        df (pandas.DataFrame): DataFrame con datos con una columna con valores 'yes', 'no'.
        columna_a_normalizar (str): String con el nombre de la columna que contiene valores 'yes' y 'no'.

    Returns:
        pandas.DataFrame: DataFrame con valores binarios en la columna 'columna_a_normalizar'.
    """
    dict_valores_a_cambiar={'yes':1,'no':0} #Creamos un dicionario.
    valores_no_contemplados = set(df[columna_a_normalizar]) - set(dict_valores_a_cambiar.keys()) # Comprobamos si hay valores no contemplados en el diccionario.
    if valores_no_contemplados:
        print(f'Existen valores no contemplados en la columna ' + columna_a_normalizar + ': {valores_no_contemplados}') # Si hay valores no contemplados, los mostramos.
    df[columna_a_normalizar] = df[columna_a_normalizar].replace(dict_valores_a_cambiar)
    df[columna_a_normalizar] = df[columna_a_normalizar].astype('int64') # Convertimos a int64.

    return df


# ## Funciones Visualizacion de Datos

# ### EDA

# #### Analisis Univariable Variables Categoricas

# Diagrama de barras para el analisis univariable

# In[8]:


def plotly_barras_univariable(df,variable):

    df = (
        df[variable]
        .value_counts()
        .rename_axis(variable)
        .reset_index(name="count")
    )

    # Crea una columna que contenga el porcentage de cada variable
    df["percentage"] = 100 * df["count"] / df["count"].sum()

    # Guarda el grafico dentro de fig
    fig = px.bar(
        data_frame = df,
        x = variable,
        y = "percentage",

        text=df["percentage"].map(lambda x: f"{x:.1f}%"),
        hover_data={"count": True, "percentage": ":.1ff"},
    )

    # Subir un poco el grafico para que se puedan observar correctamente los valores
    fig.update_yaxes(range=[0, df["percentage"].max() * 1.1])

    # Muestra el porcentaje encima de cada barra
    fig.update_traces(textposition="outside")

    # Sacar la variable texto del hover
    fig.update_traces(hovertemplate="%{x}<br>Percentage: %{y:.1f}%<br>Count: %{customdata[0]}")

    # Da informacion sobre el grafico
    fig.update_layout(
        yaxis_title="Percentage (%)",
        xaxis_title=variable.capitalize(),
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        plot_bgcolor="white"
    )

    fig.show()


# #### Analisis Univariable Variables Numericas

# Histograma para el analisis univariable

# In[9]:


def plotly_histograma_univariable(df,variable):

    df = (
        df[variable]
        .value_counts()
        .reset_index()
    )

    fig = px.histogram(
        data_frame = df,
        x = variable,
        histnorm="percent",
        text_auto=".2f",
        hover_data={"count": True},
    )

    # Muestra el porcentaje encima de cada barra
    fig.update_traces(textposition="outside")

    # Da informacion sobre el grafico
    fig.update_layout(
        yaxis_title="Percentage (%)",
        xaxis_title=variable.capitalize(),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        plot_bgcolor="white"
    )
    fig.show()


# ## Transformar el notebook en un archivo .py

# In[1]:


get_ipython().system('jupyter nbconvert --to script funciones_data_analisis.ipynb')

