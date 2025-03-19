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
PRODUCTOS = ['Producto A', 'Producto B', 'Producto C', 'Producto D', 'Producto E']
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

# Configuración de Streamlit
st.set_page_config(page_title="Análisis de Ventas", layout="wide")
st.title("Análisis de Ventas y Predicción")

# Generar datos de ventas
df_ventas = generar_datos_ventas(NUM_REGISTROS, PRODUCTOS, REGIONES, FECHA_INICIO, FECHA_FIN)

# Mostrar datos en un menú desplegable
with st.expander("Ver Datos de Ventas Generados"):
    st.write(df_ventas)

