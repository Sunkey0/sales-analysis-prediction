import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import plotly.express as px
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

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

def predecir_con_arima(serie_temporal, orden=(1, 1, 1), pasos=6):
    """
    Realiza predicciones usando un modelo ARIMA.
    """
    modelo = ARIMA(serie_temporal, order=orden)
    modelo_ajustado = modelo.fit()
    predicciones_futuras = modelo_ajustado.forecast(steps=pasos)
    predicciones_historicas = modelo_ajustado.fittedvalues
    return predicciones_futuras, predicciones_historicas, modelo_ajustado

def predecir_con_prophet(df, periodo_prediccion=6):
    """
    Realiza predicciones usando Facebook Prophet.
    """
    modelo = Prophet()
    modelo.fit(df)
    future = modelo.make_future_dataframe(periods=periodo_prediccion, freq='M')
    predicciones = modelo.predict(future)
    return predicciones, modelo

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

# Convertir la columna 'Mes' a formato de fecha (datetime)
ventas_por_mes['Mes'] = ventas_por_mes['Mes'].dt.to_timestamp()

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

# Mapa de calor de correlaciones
st.header("Mapa de Calor de Correlaciones")
correlaciones = df_filtrado[['Cantidad Vendida', 'Precio Unitario', 'Costo Unitario', 'Ingreso', 'Beneficio']].corr()
fig = px.imshow(correlaciones, text_auto=True, title='Mapa de Calor de Correlaciones')
st.plotly_chart(fig, use_container_width=True)

# Predicción con ARIMA
st.header("Predicción de Ventas con ARIMA")
ventas_por_mes.set_index('Mes', inplace=True)
predicciones_futuras_arima, predicciones_historicas_arima, modelo_arima = predecir_con_arima(ventas_por_mes['Ingreso'], orden=(1, 1, 1))

# Mostrar predicciones futuras en un menú desplegable
with st.expander("Predicciones ARIMA para los próximos 6 meses"):
    st.write(predicciones_futuras_arima)

# Mostrar predicciones históricas en un menú desplegable
with st.expander("Predicciones ARIMA dentro de la muestra (histórico)"):
    st.write(predicciones_historicas_arima)

# Calcular el error absoluto medio (MAE) para ARIMA
mae_arima = mean_absolute_error(ventas_por_mes['Ingreso'], predicciones_historicas_arima)
st.write(f"Error Absoluto Medio (MAE) para ARIMA: {mae_arima:.2f}")

# Predicción con Prophet
st.header("Predicción de Ventas con Facebook Prophet")
df_prophet = ventas_por_mes.reset_index()[['Mes', 'Ingreso']]
df_prophet.columns = ['ds', 'y']

# Asegurarse de que 'ds' sea de tipo datetime
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

predicciones_prophet, modelo_prophet = predecir_con_prophet(df_prophet)

# Mostrar predicciones futuras en un menú desplegable
with st.expander("Predicciones Prophet para los próximos 6 meses"):
    st.write(predicciones_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))

# Mostrar predicciones históricas en un menú desplegable
with st.expander("Predicciones Prophet dentro de la muestra (histórico)"):
    st.write(predicciones_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(len(df_prophet)))

# Gráfico comparativo de predicciones
st.header("Comparación de Predicciones: ARIMA vs Prophet")
fig = px.line()
fig.add_scatter(x=ventas_por_mes.index, y=ventas_por_mes['Ingreso'], name='Datos Reales', mode='lines+markers')
fig.add_scatter(x=predicciones_historicas_arima.index, y=predicciones_historicas_arima, name='Predicción ARIMA (Histórico)', mode='lines+markers')
fig.add_scatter(x=predicciones_futuras_arima.index, y=predicciones_futuras_arima, name='Predicción ARIMA (Futuro)', mode='lines+markers')
fig.add_scatter(x=predicciones_prophet['ds'], y=predicciones_prophet['yhat'], name='Predicción Prophet', mode='lines+markers')
fig.update_layout(title='Comparación de Predicciones: ARIMA vs Prophet',
                  xaxis_title='Mes', yaxis_title='Ingreso ($)')
st.plotly_chart(fig, use_container_width=True)

# Sección de KPIs y Alarmas
st.header("KPIs y Alarmas")

# Calcular KPIs
ingreso_total = df_filtrado['Ingreso'].sum()
beneficio_total = df_filtrado['Beneficio'].sum()
crecimiento_mensual = ventas_por_mes['Ingreso'].pct_change().iloc[-1] * 100

# Mostrar KPIs
st.subheader("KPIs Clave")
col1, col2, col3 = st.columns(3)
col1.metric("Ingreso Total", f"${ingreso_total:,.2f}")
col2.metric("Beneficio Total", f"${beneficio_total:,.2f}")
col3.metric("Crecimiento Mensual", f"{crecimiento_mensual:.2f}%")

# Alarmas
st.subheader("Alarmas")
if beneficio_total < 100000:
    st.error("⚠️ El beneficio total es menor a $100,000. ¡Revisar estrategias!")
if crecimiento_mensual < 0:
    st.warning("⚠️ El crecimiento mensual es negativo. ¡Investigar causas!")

# Información Relevante
st.header("Información Relevante")

# Restablecer el índice para acceder a la columna 'Mes'
ventas_por_mes_reset = ventas_por_mes.reset_index()

# Mostrar el mes con mayores ingresos
mes_max_ingresos = ventas_por_mes_reset.loc[ventas_por_mes_reset['Ingreso'].idxmax()]['Mes']
st.write("**Mes con mayores ingresos:**", mes_max_ingresos.strftime('%Y-%m'))

# Mostrar el producto más rentable
producto_mas_rentable = ventas_por_producto.loc[ventas_por_producto['Beneficio'].idxmax()]['Producto']
st.write("**Producto más rentable:**", producto_mas_rentable)

# Mostrar la región con mayor crecimiento
region_mayor_crecimiento = ventas_por_region.loc[ventas_por_region['Ingreso'].idxmax()]['Región']
st.write("**Región con mayor crecimiento:**", region_mayor_crecimiento)

# Sección de Ecuaciones y Fórmulas
st.header("Ecuaciones y Fórmulas Utilizadas")

st.markdown("""
### 1. Ingreso
El ingreso se calcula multiplicando la cantidad vendida por el precio unitario:
""")
st.latex(r"\text{Ingreso} = \text{Cantidad Vendida} \times \text{Precio Unitario}")

st.markdown("""
### 2. Beneficio
El beneficio se calcula restando los costos totales del ingreso:
""")
st.latex(r"\text{Beneficio} = \text{Ingreso} - (\text{Cantidad Vendida} \times \text{Costo Unitario})")

st.markdown("""
### 3. Crecimiento Mensual
El crecimiento mensual se calcula como el cambio porcentual en los ingresos de un mes a otro:
""")
st.latex(r"\text{Crecimiento Mensual} = \frac{\text{Ingreso del Mes Actual} - \text{Ingreso del Mes Anterior}}{\text{Ingreso del Mes Anterior}} \times 100")

st.markdown("""
### 4. Error Absoluto Medio (MAE)
El MAE mide la precisión de las predicciones comparando los valores reales con los predichos:
""")
st.latex(r"\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|")

st.markdown("""
### 5. Modelo ARIMA
El modelo ARIMA se define por tres parámetros: \( p \) (términos autorregresivos), \( d \) (diferenciaciones) y \( q \) (términos de media móvil):
""")
st.latex(r"\text{ARIMA}(p, d, q)")

st.markdown("""
### 6. Modelo Prophet
El modelo Prophet descompone la serie temporal en tendencia (\( g(t) \)), estacionalidad (\( s(t) \)), días festivos (\( h(t) \)) y ruido (\( \epsilon_t \)):
""")
st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
