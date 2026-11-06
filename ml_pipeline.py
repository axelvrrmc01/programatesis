import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def preparar_datos_historicos(df):
    df['fecha_y_hora_de_pedido'] = pd.to_datetime(df['fecha_y_hora_de_pedido'])
    df['mes_año'] = df['fecha_y_hora_de_pedido'].dt.to_period('M')
    df['mes'] = df['fecha_y_hora_de_pedido'].dt.month
    return df

def formatear_mes_anio(periodo_str):
    meses = {
        '01':'Marzo', '02':'Febrero', '03':'Marzo', '04':'Abril', 
        '05':'Mayo', '06':'Junio', '07':'Julio', '08':'Agosto', 
        '09':'Septiembre', '10':'Octubre', '11':'Noviembre', '12':'Diciembre'
    }
    # En caso de que haya un error de mapeo, reparamos Enero
    meses['01'] = 'Enero'
    
    año, mes = periodo_str.split('-')
    return f"{meses[mes]} {año}"

def obtener_top_3_por_mes(df):
    df = preparar_datos_historicos(df)
    
    df_mensual = df.groupby(['mes_año', 'descripcion'])['cantidad'].sum().reset_index()
    df_ordenado = df_mensual.sort_values(['mes_año', 'cantidad'], ascending=[True, False])
    top_3 = df_ordenado.groupby('mes_año').head(3).copy()
    
    # Formatear la columna al estilo "Marzo 2024"
    top_3['mes_año'] = top_3['mes_año'].astype(str)
    top_3['Periodo'] = top_3['mes_año'].apply(formatear_mes_anio)
    
    # Renombrar columnas
    top_3 = top_3.rename(columns={'descripcion': 'Producto', 'cantidad': 'Cantidad'})
    
    return top_3[['Periodo', 'Producto', 'Cantidad']]

def entrenar_y_predecir(df):
    df = preparar_datos_historicos(df)
    
    # 1. Preparación
    df_hist = df.groupby(['codigo_de_producto', 'descripcion', 'mes'])['cantidad'].sum().reset_index()
    media_ventas = df.groupby('codigo_de_producto')['cantidad'].mean().reset_index()
    media_ventas.rename(columns={'cantidad': 'media_historica'}, inplace=True)
    
    df_ml = pd.merge(df_hist, media_ventas, on='codigo_de_producto')
    
    X = df_ml[['mes', 'media_historica']]
    y = df_ml['cantidad']
    
    # 2. Entrenamiento
    modelo = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    modelo.fit(X, y)
    
    # 3. Base para predicciones futuras
    productos_unicos = df[['codigo_de_producto', 'descripcion']].drop_duplicates()
    meses_futuros = list(range(1, 13))
    
    futuro = productos_unicos.assign(key=1).merge(pd.DataFrame({'mes': meses_futuros, 'key': 1}), on='key').drop('key', axis=1)
    futuro = pd.merge(futuro, media_ventas, on='codigo_de_producto', how='left').fillna(0)
    
    # 4. Inferencia
    X_futuro = futuro[['mes', 'media_historica']]
    futuro['Cantidad Proyectada'] = modelo.predict(X_futuro)
    futuro['Cantidad Proyectada'] = futuro['Cantidad Proyectada'].clip(lower=0).round().astype(int)
    
    # Renombrar descripción
    futuro = futuro.rename(columns={'descripcion': 'Producto'})
    
    return futuro[['mes', 'Producto', 'Cantidad Proyectada']]
