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


