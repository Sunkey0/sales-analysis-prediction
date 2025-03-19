import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import plotly.express as px
import streamlit as st

# Configuración
NUM_REGISTROS = 10000
PRODUCTOS = ['Producto A', 'Producto B', 'Producto C', 'Producto D', 'Producto E', 
             'Producto F', 'Producto G', 'Producto H', 'Producto I', 'Producto J']
REGIONES = ['Norte', 'Sur', 'Este', 'Oeste']
FECHA_INICIO = datetime(2023, 1, 1)
FECHA_FIN = datetime(2023, 12, 31)

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

# Filtros
st.sidebar.header("Filtros")
fecha_min = df_ventas['Fecha'].min()
fecha_max = df_ventas['Fecha'].max()
rango_fechas = st.sidebar.date_input(
    "Selecciona el rango de fechas",
    [fecha_min, fecha_max],
    min_value=fecha_min,
    max_value=fecha_max
)

productos_seleccionados = st.sidebar.multiselect(
    "Selecciona los productos",
    PRODUCTOS,
    default=PRODUCTOS
)

regiones_seleccionadas = st.sidebar.multiselect(
    "Selecciona las regiones",
    REGIONES,
    default=REGIONES
)

# Aplicar filtros
df_filtrado = df_ventas[
    (df_ventas['Fecha'] >= pd.to_datetime(rango_fechas[0])) &
    (df_ventas['Fecha'] <= pd.to_datetime(rango_fechas[1])) &
    (df_ventas['Producto'].isin(productos_seleccionados)) &
    (df_ventas['Región'].isin(regiones_seleccionadas))
]

# Análisis de ventas por producto y región
st.header("Análisis de Ventas por Producto y Región")
ventas_por_producto = df_filtrado.groupby('Producto').agg({
    'Ingreso': 'sum',
    'Beneficio': 'sum',
    'Cantidad Vendida': 'sum'
}).reset_index()

ventas_por_region = df_filtrado.groupby('Región').agg({
    'Ingreso': 'sum',
    'Beneficio': 'sum',
    'Cantidad Vendida': 'sum'
}).reset_index()

# Mostrar DataFrames en menús desplegables
with st.expander("Ventas por Producto"):
    st.write(ventas_por_producto)

with st.expander("Ventas por Región"):
    st.write(ventas_por_region)

# Análisis de ventas mensuales
st.header("Análisis de Ventas Mensuales")
df_filtrado['Mes'] = df_filtrado['Fecha'].dt.to_period('M')
ventas_por_mes = df_filtrado.groupby('Mes').agg({
    'Ingreso': 'sum',
    'Beneficio': 'sum',
    'Cantidad Vendida': 'sum'
}).reset_index()
ventas_por_mes['Mes'] = ventas_por_mes['Mes'].astype(str)

# Mostrar DataFrame en un menú desplegable
with st.expander("Ventas Mensuales"):
    st.write(ventas_por_mes)

# Gráficos de ingresos y beneficios mensuales
st.header("Gráficos de Ingresos y Beneficios Mensuales")
fig = px.line(ventas_por_mes, x='Mes', y=['Ingreso', 'Beneficio'], 
              title='Ingresos y Beneficios Mensuales', labels={'value': 'Monto ($)'})
st.plotly_chart(fig, use_container_width=True)

# Gráfico de ventas por producto
st.header("Ventas por Producto")
fig = px.bar(ventas_por_producto, x='Producto', y='Ingreso', 
             title='Ingresos por Producto', labels={'Ingreso': 'Ingreso ($)'})
st.plotly_chart(fig, use_container_width=True)

# Gráfico de ventas por región
st.header("Ventas por Región")
fig = px.bar(ventas_por_region, x='Región', y='Ingreso', 
             title='Ingresos por Región', labels={'Ingreso': 'Ingreso ($)'})
st.plotly_chart(fig, use_container_width=True)

# Gráfico de dispersión: Ingreso vs Beneficio
st.header("Relación entre Ingreso y Beneficio")
fig = px.scatter(df_filtrado, x='Ingreso', y='Beneficio', color='Producto', 
                 title='Ingreso vs Beneficio', labels={'Ingreso': 'Ingreso ($)', 'Beneficio': 'Beneficio ($)'})
st.plotly_chart(fig, use_container_width=True)
