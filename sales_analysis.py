import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import streamlit as st

# Configuración
NUM_REGISTROS = 10000
PRODUCTOS = ['Producto A', 'Producto B', 'Producto C', 'Producto D', 'Producto E', 
             'Producto F', 'Producto G', 'Producto H', 'Producto I', 'Producto J']
REGIONES = ['Norte', 'Sur', 'Este', 'Oeste']
FECHA_INICIO = datetime(2023, 1, 1)
FECHA_FIN = datetime(2023, 12, 31)

# Estilo de gráficos
sns.set(style="whitegrid")

def generar_datos_ventas(num_registros, productos, regiones, fecha_inicio, fecha_fin):
    """
    Genera un DataFrame con datos de ventas aleatorios.

    Args:
        num_registros (int): Número de registros a generar.
        productos (list): Lista de nombres de productos.
        regiones (list): Lista de nombres de regiones.
        fecha_inicio (datetime): Fecha inicial para las ventas.
        fecha_fin (datetime): Fecha final para las ventas.

    Returns:
        pd.DataFrame: DataFrame con datos de ventas.
    """
    fake = Faker()
    datos = {
        'Fecha': [fecha_inicio + timedelta(days=random.randint(0, (fecha_fin - fecha_inicio).days)) 
                  for _ in range(num_registros)],
        'Producto': [random.choice(productos) for _ in range(num_registros)],
        'Región': [random.choice(regiones) for _ in range(num_registros)],
        'Cantidad Vendida': [random.randint(1, 100) for _ in range(num_registros)],
        'Precio Unitario': [round(random.uniform(10, 100), 2) for _ in range(num_registros)],
        'Costo Unitario': [round(random.uniform(5, 50), 2) for _ in range(num_registros)]
    }
    df = pd.DataFrame(datos)
    df['Ingreso'] = df['Cantidad Vendida'] * df['Precio Unitario']
    df['Beneficio'] = df['Ingreso'] - (df['Cantidad Vendida'] * df['Costo Unitario'])
    return df

def analizar_ventas_por(df, columna_agrupacion):
    """
    Agrupa los datos de ventas por una columna específica y calcula métricas agregadas.

    Args:
        df (pd.DataFrame): DataFrame con datos de ventas.
        columna_agrupacion (str): Columna por la cual agrupar.

    Returns:
        pd.DataFrame: DataFrame con métricas agregadas.
    """
    return df.groupby(columna_agrupacion).agg({
        'Ingreso': 'sum',
        'Beneficio': 'sum',
        'Cantidad Vendida': 'sum'
    }).reset_index()


def calcular_crecimiento_mensual(df):
    """
    Calcula el crecimiento mensual de ingresos y beneficios.

    Args:
        df (pd.DataFrame): DataFrame con datos de ventas mensuales.

    Returns:
        pd.DataFrame: DataFrame con crecimiento mensual.
    """
    df['Crecimiento Ingreso'] = df['Ingreso'].pct_change() * 100
    df['Crecimiento Beneficio'] = df['Beneficio'].pct_change() * 100
    return df

# Configuración de Streamlit
st.set_page_config(page_title="Análisis de Ventas", layout="wide")
st.title("Análisis de Ventas y Predicción")

# Generar datos de ventas
df_ventas = generar_datos_ventas(NUM_REGISTROS, PRODUCTOS, REGIONES, FECHA_INICIO, FECHA_FIN)

# Mostrar datos en un menú desplegable
with st.expander("Ver Datos de Ventas Generados"):
    st.write(df_ventas)

# Análisis de ventas por producto y región
st.header("Análisis de Ventas por Producto y Región")
ventas_por_producto = analizar_ventas_por(df_ventas, 'Producto')
ventas_por_region = analizar_ventas_por(df_ventas, 'Región')

# Mostrar DataFrames en menús desplegables
with st.expander("Ventas por Producto"):
    st.write(ventas_por_producto)

with st.expander("Ventas por Región"):
    st.write(ventas_por_region)

# Análisis de ventas mensuales
st.header("Análisis de Ventas Mensuales")
df_ventas['Mes'] = df_ventas['Fecha'].dt.to_period('M')
ventas_por_mes = analizar_ventas_por(df_ventas, 'Mes')
ventas_por_mes = calcular_crecimiento_mensual(ventas_por_mes)

# Mostrar DataFrame en un menú desplegable
with st.expander("Ventas Mensuales"):
    st.write(ventas_por_mes)

# Gráficos de ingresos y beneficios mensuales
st.header("Gráficos de Ingresos y Beneficios Mensuales")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ventas_por_mes['Mes'].astype(str), ventas_por_mes['Ingreso'], label='Ingreso', marker='o')
ax.plot(ventas_por_mes['Mes'].astype(str), ventas_por_mes['Beneficio'], label='Beneficio', marker='o')
ax.set_title('Ingresos y Beneficios Mensuales')
ax.set_xlabel('Mes')
ax.set_ylabel('Monto ($)')
ax.legend()
st.pyplot(fig)


