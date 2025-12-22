# Librerías estándar
import os
import warnings

# Manipulación de datos
import pandas as pd
import numpy as np

# Visualización de datos
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Análisis de nulos
import missingno as msno

# Estadística
import scipy.stats as stats

# Configuración de warnings
warnings.filterwarnings('ignore')

def leer_archivo(ruta_completa):
    try:

        _, extension = os.path.splitext(ruta_completa.lower())


        if extension == '.csv':
            df = pd.read_csv(ruta_completa)
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




def exploracion_inicial(df, nombre=None, tipo=None):
    """
    Realiza una exploración inicial de un DataFrame y muestra información clave.

    Parámetros:
    df (pd.DataFrame): El DataFrame a explorar.
    tipo (str, opcional): El tipo de exploración. 'simple' muestra menos detalles.

    Imprime:
    Información relevante sobre el DataFrame, incluyendo filas, columnas, tipos de datos,
    estadísticas descriptivas, y valores nulos.
    """
    if nombre:
      print(nombre.upper().center(90, ' # '))
      print('\n\n')

    # Información básica sobre el DataFrame
    num_filas, num_columnas = df.shape
    print(f"¿Cuántas filas y columnas hay en el conjunto de datos?")
    print(f"\tHay {num_filas:,} filas y {num_columnas:,} columnas.")
    print('#' * 90)

    # Exploración simple
    if tipo == 'simple':
        print("¿Cuáles son las primeras dos filas del conjunto de datos?")
        display(df.head(2))
    else:
        # Exploración completa
        print("¿Cuáles son las primeras cinco filas del conjunto de datos?")
        display(df.head())
        print('-' * 100)

        print("¿Cuáles son las últimas cinco filas del conjunto de datos?")
        display(df.tail())
        print('-' * 100)

        print("¿Cómo puedes obtener una muestra aleatoria de filas del conjunto de datos?")
        display(df.sample(n=5))
        print('-' * 100)

        print("¿Cuáles son las columnas del conjunto de datos?")
        print("\n".join(f"\t- {col}" for col in df.columns))
        print('-' * 100)

        print("¿Cuál es el tipo de datos de cada columna?")
        print(df.dtypes)
        print('-' * 100)

        print("¿Cuántas columnas hay de cada tipo de datos?")
        print(df.dtypes.value_counts())
        print('-' * 100)

        print("¿Cómo podríamos obtener información más completa sobre la estructura y el contenido del DataFrame?")
        print(df.info())
        print('-' * 100)

        print("¿Cuántos valores únicos tiene cada columna?")
        print(df.nunique())
        print('-' * 100)

        print("¿Cuáles son los valores únicos de cada columna?")
        df_valores_unicos = pd.DataFrame(df.apply(lambda x: x.unique()))
        display(df_valores_unicos)
        print('-' * 100)

        print("¿Cuáles son las estadísticas descriptivas básicas de todas las columnas?")
        display(df.describe(include='all').fillna(''))
        print('-' * 100)

        print("¿Cuántos valores nulos hay en cada columna del DataFrame?")
        display(df.isnull().sum())
        print('-' * 100)

        print("¿Cuál es el porcentaje de valores nulos por columna, ordenado de mayor a menor?")
        df_nulos = df.isnull().sum().div(len(df)).mul(100).round(2).reset_index().rename(columns = {'index': 'Col', 0: 'pct'})
        df_nulos = df_nulos.sort_values(by = 'pct', ascending=False).reset_index(drop = True)
        display(df_nulos)
        print('-' * 100)

        print("## Valores nulos: Visualización")
        msno.bar(df, figsize = (6, 3), fontsize= 9)
        plt.show()
        print('-' * 100)

        print("## Visualización de patrones en valores nulos")
        msno.matrix(df, figsize = (6, 3), fontsize= 9, sparkline = False)
        plt.show()
        print('-' * 100)

        msno.heatmap(df, figsize = (6, 3), fontsize= 9)
        plt.show()
        print('-' * 100)

    print('#' * 90)


def graficar_boxplot_px(df, variable_analisis):
     # Crear el boxplot usando Plotly Express
     fig = px.box(df, y=variable_analisis)

     # Actualizar títulos del gráfico
     fig.update_layout(title=f'Boxplot: {variable_analisis}',
                       yaxis_title='Frecuencia',
         width=600,     # ancho en píxeles
         height=400     # alto en píxeles
                       )

     # Actualizar el fondo del gráfico a blanco
     fig.update_layout({
         'plot_bgcolor': 'rgba(255, 255, 255, 1)',
         'xaxis': {'showgrid': True, 'gridcolor': 'lightgrey'},
         'yaxis': {'showgrid': True, 'gridcolor': 'lightgrey'}
     })

     # Mostrar el gráfico
     fig.show()


def deteccion_outliers (df, variable):
    columna = df[variable]

    sns.boxplot(
      data=df,
      y=variable,
    )
    plt.show()

    Q1 = columna.quantile(0.25)
    Q3 = columna.quantile(0.75)
    IQR = Q3 - Q1

    print('Valor del segundo cuartil (25%): {:.2f}'.format(Q1))
    print('Valor del tercer cuartil (75%): {:.2f}'.format(Q3))
    print('Valor del rango intercuartil (IQR): {:.2f}'.format(IQR))

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    print(f"Los valores atípicos se definen como aquellos que caen fuera del siguiente rango:")
    print(f"\t - Límite inferior (considerado extremadamente bajo): {limite_inferior:.2f}")
    print(f"\t - Límite superior (considerado extremadamente alto): {limite_superior:.2f}")

    outilers = list(columna[((columna < limite_inferior) | (columna > limite_superior))].index)
    num_outliers = len(outilers)
    print(f"Hay {num_outliers} outliers en la variable '{variable}'")
    return outilers


def graficar_barras_px (df, variable_analisis):
    # Contar la frecuencia de la variable de análisis
    volumen = df[variable_analisis].value_counts().reset_index()
    volumen.columns = [variable_analisis, 'Volumen']

    # Crear el gráfico de barras
    fig = px.bar(volumen, x=variable_analisis, y='Volumen', text='Volumen')
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(title_text=f'Gráfico de barras: {variable_analisis}',
                      xaxis_title=variable_analisis,
                      yaxis_title='Volumen',
                      xaxis={'categoryorder':'total descending'})

    # Actualizar el fondo del gráfico a blanco
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'xaxis': {'showgrid': True, 'gridcolor': 'lightgrey'},
        'yaxis': {'showgrid': True, 'gridcolor': 'lightgrey'}
    })

    fig.show()


def graficar_barras_relativo_px(df, variable_analisis):
    """
    Definición:
    La función graficar_barras_relativo_px genera un gráfico de barras utilizando la librería Plotly Express para visualizar la frecuencia relativa (porcentaje) de una variable categórica en un DataFrame.

    Parámetros:
    - df: El DataFrame que contiene los datos.
    - variable_analisis: El nombre de la columna en el DataFrame cuya frecuencia relativa se desea representar en el gráfico de barras.

    Utilidad:
    Esta función permite crear gráficos de barras que muestran la distribución porcentual de una variable categórica, facilitando la comparación visual entre categorías. El gráfico incluye etiquetas con los valores absolutos y relativos, mejorando la interpretación de los datos y ayudando a identificar tendencias en la distribución de la variable analizada.
    """

    # Calcular la frecuencia absoluta y relativa de la variable de análisis
    volumen = df[variable_analisis].value_counts(normalize=False).reset_index()
    volumen.columns = [variable_analisis, 'Frecuencia']

    # Calcular el porcentaje relativo
    volumen['Porcentaje'] = df[variable_analisis].value_counts(normalize=True).reset_index(drop=True) * 100

    # Crear el texto con valores absolutos y relativos
    volumen['Texto'] = volumen.apply(lambda row: f'{row["Frecuencia"]} ({row["Porcentaje"]:.2f}%)', axis=1)

    # Crear el gráfico de barras
    fig = px.bar(volumen, x=variable_analisis, y='Porcentaje', text=volumen['Texto'])
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(title_text=f'Gráfico de barras relativo: {variable_analisis}',
                      xaxis_title=variable_analisis,
                      yaxis_title='Porcentaje',
                      xaxis={'categoryorder': 'total descending'})

    # Actualizar el fondo del gráfico a blanco
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'xaxis': {'showgrid': True, 'gridcolor': 'lightgrey'},
        'yaxis': {'showgrid': True, 'gridcolor': 'lightgrey'}
    })

    fig.show()

def graficar_histograma_px (df, variable_analisis):
    fig = px.histogram(df, x=variable_analisis, nbins=20,
                       title=f'Distribución de: {variable_analisis}')

    # Calcular media y mediana
    mean_val = df[variable_analisis].mean()
    median_val = df[variable_analisis].median()

    # Añadir línea vertical para la media
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Media: {mean_val:.2f}", annotation_position="top right")

    # Añadir línea vertical para la mediana
    fig.add_vline(x=median_val, line_dash="dot", line_color="green",
                  annotation_text=f"Mediana: {median_val:.2f}", annotation_position="top left")

    fig.update_layout(xaxis_title=variable_analisis, yaxis_title='Frecuencia')
    fig.show()


def derretir_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    """
    Reestructura un df de formato ancho a formato largo.

    Args:
        df: El df de entrada que se desea transformar.

    Returns:
        df: El DataFrame transformado (largo). IMP reasignar el df.
    """
    df = df.melt(
        id_vars=["Eventos", "Año", "Genero"],        
        value_vars=["Oro", "Plata", "Bronce"],       
        var_name="Medalla",                         
        value_name="Resultado"                      
    )
    return df




def extraccion_texto(df: pd.DataFrame, columna_origen: str, patron: str, columna_destino: str) -> pd.DataFrame:
    """
    Extrae un patrón de texto usando una expresión regular y guarda el resultado 
    en una nueva columna del df. Posteriormente limpia la columna de origen.

    Args:
        df: El df de entrada.
        columna_origen: La columna de donde se extraerá el texto.
        patrón: La expresión regular a aplicar.
        columna_destino: El nombre de la nueva columna para guardar el resultado.

    Returns:
        Devuelve el df modificado (no hace falta reasignar)
    """
    if columna_origen not in df.columns:
        print(f"Error: La columna de origen {columna_origen} no está en el df.")
        return df
    
    else:
        df[columna_destino] = df[columna_origen].astype(str).str.extract(patron, expand=False)
        df[columna_origen] = df[columna_origen].astype(str).str.replace(patron,"",regex=True).str.strip()
        
        return df
