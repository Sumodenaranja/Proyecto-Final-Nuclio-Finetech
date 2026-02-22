#!/usr/bin/env python
# coding: utf-8

# # Funciones FinTech

# Estas funciones son funciones necesarias para realizar el proyecto de FinTech

# ## Importacion de librerias

# In[ ]:


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
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter
from sklearn.feature_selection import mutual_info_classif


# Textos
import unicodedata
from fuzzywuzzy import process
import re

# Estadistica
from scipy import stats
from scipy.stats import chi2_contingency
from itertools import combinations



# ## Funciones limpieza y preparacion de datos

# ### Funcion para renombrar columnas originales

# In[ ]:


def renombrar_columnas(df):
    nombre_columnas = ['age', 'job', 'marital_status', 'education', 'credit_default', 'housing_loan', 'personal_loan',
       'contact_type', 'last_contact_month', 'last_contact_day', 'last_contact_duration_secs', 
       'number_contacts', 'number_days_last_contact','number_of_previous_contacts', 
       'outcome_previous_campaign', 'employement_variation_rate', 'consumer_price_index','consumer_confidence_index', 
       'euribor_3m', 'number_employees', 'subscribed_term_deposit', 'age_group',
       'last_contact_duration_mins', 'last_contact_duration_mins_group',
       'employement_variation_rate_group','consumer_price_index_group','consumer_confidence_index_group','euribor_3m_group']
    df.columns = nombre_columnas

    return df


# ### Funcion convertir columnas a tipo categórico.

# In[3]:


def columnas_categoricas(df):
    # Listas categorias.
    lista_categorias_job=['unknown','unemployed','student','retired','housemaid', 'services','blue_collar',
                        'self_employed',  'administrative_staff',  'technician','entrepreneur','management']

    lista_marital_status=['unknown','single','married','divorced']

    lista_categorias_education=['unknown','illiterate','basic_4y','basic_6y', 'basic_9y', 
                                'high_school', 'professional_course',  'university_degree']

    lista_credit_default=['unknown','yes','no']

    lista_housing_loan=['unknown','yes','no']

    lista_personal_loan=['unknown','yes','no']

    lista_contact_type=['telephone','cellular']

    lista_last_contact_month=['mar', 'apr','may', 'jun', 'jul', 'aug','sep', 'oct', 'nov', 'dec']

    lista_last_contact_day=['mon', 'tue', 'wed', 'thu', 'fri']

    lista_outcome_previous_campaign=['nonexistent', 'failure', 'success']

    # Convertir columnas a tipo categórico segun las listas definidas.
    df['job'] = pd.Categorical(df['job'], categories=lista_categorias_job, ordered=False)
    df['marital_status'] = pd.Categorical(df['marital_status'], categories=lista_marital_status, ordered=False)
    df['education'] = pd.Categorical(df['education'], categories=lista_categorias_education, ordered=False)
    df['credit_default'] = pd.Categorical(df['credit_default'], categories=lista_credit_default, ordered=False)
    df['housing_loan'] = pd.Categorical(df['housing_loan'], categories=lista_housing_loan, ordered=False)
    df['personal_loan'] = pd.Categorical(df['personal_loan'], categories=lista_personal_loan, ordered=False)
    df['contact_type'] = pd.Categorical(df['contact_type'], categories=lista_contact_type, ordered=False)
    df['last_contact_month'] = pd.Categorical(df['last_contact_month'], categories=lista_last_contact_month, ordered=True)
    df['last_contact_day'] = pd.Categorical(df['last_contact_day'], categories=lista_last_contact_day, ordered=False)
    df['outcome_previous_campaign'] = pd.Categorical(df['outcome_previous_campaign'], categories=lista_outcome_previous_campaign, ordered=False)

    return df


# ### Funcion para normalizar textos

# In[ ]:


def normalizar_textos(df):
    for col in df.select_dtypes(include=['object']).columns: # Iteramos sobre columnas de tipo object.
        df[col]=df[col].str.lower() # Convertimos a minúsculas.
        df[col]=df[col].str.strip() # Eliminamos espacios en blanco al inicio y final.
        df[col]=df[col].str.replace(r'\.+$', '', regex=True) # Eliminamos los '.' al final de las cadenas de texto.
        df[col]=df[col].str.replace(r'(?<=\w)\.+(?=\w)', '_', regex=True) # Reemplazamos los '.' entre palabras por '_'.
        df[col]=df[col].str.replace(r'(?<=\w)-+(?=\w)', '_', regex=True) # Reemplazamos los '-' entre palabras por '_'.
        dict_job={'admin':'administrative_staff'}
        if col=='job':
            df[col]=df[col].replace(dict_job)

    return df


# ### Funcion cambiar columna subscribed_term_deposit a binario

# In[ ]:


def normalizar_binario(df):
    dict_subscribed_term_deposit={'yes':1,'no':0} #Creamos un dicionario.
    nulos=set(df.subscribed_term_deposit)-set(dict_subscribed_term_deposit.keys()) # Comprobamos si hay valores no contemplados en el diccionario.
    if nulos:
        print(f'Existen valores nulos en la columna subscribed_term_deposit: {nulos}') # Si hay valores no contemplados, los mostramos.
    df.subscribed_term_deposit=df.subscribed_term_deposit.replace(dict_subscribed_term_deposit)
    df.subscribed_term_deposit = df.subscribed_term_deposit.astype('int64') # Convertimos a int64.


# ## Funciones graficos y obtencion de datos

# In[ ]:


def cramers_v(df):

    chi2, p, dof, expected = chi2_contingency(df)
    n = df.values.sum()              
    k = min(df.shape) - 1            

    return np.sqrt(chi2 / (n * k))


# In[ ]:


def heatmap_correlation(df, df_2, columna, title, ylable):

    # Calculamos chi2, p, dof
    chi2, p, dof, expected = stats.chi2_contingency(df)

    # Calculamos V de Cramer
    V = cramers_v(df)

    # Creamos la figura.
    plt.figure(figsize=(10, 6))

    # Normalizamos los datos para obtener el porcentaje sobre el total de filas.
    df_heatmap = df. div(df.sum(axis=1), axis=0)

    # Creamos heatmap
    ax = sns.heatmap(
        df_heatmap,
        annot=True,
        fmt='.2%',
        cmap='YlGnBu',
        cbar_kws={'label': 'Porcentaje', 'format': PercentFormatter(1)},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel(ylable, fontsize=12)

    # Conteos y porcentajes
    df_values = df_2[columna]. value_counts().to_frame().transpose()
    df_values_norm = (df_2[columna].value_counts(normalize=True) * 100).to_frame().transpose()

    # Imprimimos resultados
    print(f"chi2 = {chi2:.4f}, p = {p:.4f}, dof = {dof}")
    print(f"V de Cramer = {V:.5f}")
    print("*" * 80)
    display(df_values)
    print("*" * 80)
    display(df_values_norm)
    print("*" * 80)
    display(df)

    plt.tight_layout()
    plt.show()


# In[ ]:


# Funcion para crear un histograma de barras para la correlacion de Spearman.
def grafico_correlacion_spearman(df,df_2,title):

    # Creamos la figura
    fig, ax = plt.subplots(figsize=(6, 4))

    # Extraemos los datos del dataframe
    x_positions = df.index 
    y_values = df['subscribed_term_deposit'] 

    # Creamos el grafico de barras
    bars = ax.bar(
        x=x_positions,      
        height=y_values,   
        color="coral",
        edgecolor="black",
        alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Correlación", fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.get_xticklabels(), ha='right')
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Creamos un conteo de los indices del dataframe a graficar sobre el dataframe original
    for col in df.index:
        df_values=df_2[col].value_counts().to_frame().transpose()
        df_values_norm=(df_2[col].value_counts(normalize=True)*100).to_frame().transpose()
        display(df_values)
        display(df_values_norm)
        print("*" * 80)
    display(df)
    print("*" * 80)
    plt.tight_layout()
    plt.show()


# In[ ]:


def generar_combinaciones_de_dos(elementos):
    return list(combinations(elementos,2))


# In[ ]:


def evaluar_informacion_mutua(df, variables_a_combinar,nombres_variables=None, threshold=0.05):
    # Todos los que sean very weak o weak no nos interesaran, por eso marcaremos el threshold a 0.05
    combinaciones_relevantes = []

    for j,k in variables_a_combinar:
        df["interaccion"] = (
            df[j].astype(str) + "_" + df[k].astype(str)
        )

        X = df["interaccion"]
        y = df["subscribed_term_deposit"]

        mi = mutual_info_classif(
            pd.get_dummies(X),
            y,
            discrete_features=True
        )

        relationship_table = pd.Series(mi,index=pd.get_dummies(X).columns)
        relationship_table = relationship_table[relationship_table > threshold]

        if relationship_table.shape[0] == 0:
            # print("No hay ninguna relacion en considerable en esta interaccion")
            pass
        else:
            combinaciones_relevantes.append((j,k))
            if nombres_variables is not None:
                print(nombres_variables[j] + " x " + nombres_variables[k])
            else:
                print(j + " x " + k)

            display(relationship_table)
    return combinaciones_relevantes


# In[ ]:


def heatmap_multivariable(df, combinaciones, nombres_variables):
    for j, k in combinaciones:
        plt.figure(figsize=(10,5))

        pivot = pd.pivot_table(
            df,
            values="subscribed_term_deposit",
            index=j,
            columns=k,
            aggfunc="mean"
        )

        pivot = pivot.loc[pivot.index != "unknown", :]
        pivot = pivot.loc[:, pivot.columns != "unknown"]

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            vmin=0,
            vmax=1
        )

        plt.title(nombres_variables[j] + " x " + nombres_variables[k])
        plt.ylabel(nombres_variables[j])
        plt.xlabel(nombres_variables[k])
        plt.show()


# ## Transformar el notebook en un archivo .py

# In[3]:



