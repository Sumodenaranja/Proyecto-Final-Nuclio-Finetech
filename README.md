# Proyecto Marketing Analytics para empresa Fintech
Análisis de los datos de campañas de marketing de una empresa Fintech para conseguir identificar patrones, tendencias y factores que influyen en que un cliente acabe contratando un depósito. 

## Tareas a Realizar
### 1. Análisis de Datos
- Entendimiento de datos de clientes, de los detalles de las campañas y sus resultados.
- Estadísticas descriptivas para entender las características básicas de los datos.
- Análisis de la relación entre características demográficas del cliente (edad, trabajo, educación) y la suscripción a un depósito a plazo.
- Evaluación del impacto de los detalles de la campaña (número de contactos, mes, día de la semana) en el resultado de la campaña.
### 2. Análisis de la Respuesta de la Campaña
- Comparación de tasas de éxito entre diferentes tipos de clientes.
- Evaluar posibles indicadores del éxito de una campaña, como podría ser  la duración de la llamada
### 3. Modelado Predictivo Simple
- Construcción de un modelo de regresión logística simple para predecir la probabilidad de suscripción a un depósito a plazo basado en variables seleccionadas. Este modelo podrá implementarse directamente en Python, o utilizar cualquier otra herramienta para hacerlo (e.g. BigQueryML, chatGPT, etc.)
- Evaluación del modelo utilizando métricas básicas como precisión, sensibilidad, y especificidad. Será necesaria una familiarización previa con esas métricas a través de los recursos que se consideren oportunos.
### 4. Visualización de Datos y Creación de Dashboards
- Diseño e implementación de un dashboard interactivo utilizando la herramienta de visualización que se desee (Looker, Tableau, etc.).
- Uso de visualizaciones para destacar las tendencias en la efectividad de la campaña a lo largo del tiempo.
- Creación de gráficos para mostrar distribuciones de variables clave y principales KPIs.
### 5. Interpretación y Recomendaciones
- Identificación de patrones y factores clave que contribuyen al éxito de las campañas de marketing.
- Formulación de recomendaciones estratégicas para mejorar la efectividad de futuras campañas de marketing.

## Contexto
Las campañas de marketing se basan en llamadas telefónicas que tuvieron lugar de Mayo 2018 a Noviembre 2020. Los registros vienen ordenados por fecha, pero no hay una variable disponible con la fecha completa en el dataset original. Es importante también tener en cuenta que es posible que en algunos casos se requiera de más de un contacto con el mismo cliente para determinar si el depósito acaba siendo contratado no ('sí') o no ('no'). 
