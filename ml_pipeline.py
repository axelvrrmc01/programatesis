import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def preparar_datos_historicos(df):
    df['fecha_y_hora_de_pedido'] = pd.to_datetime(df['fecha_y_hora_de_pedido'])
    df['mes_año'] = df['fecha_y_hora_de_pedido'].dt.to_period('M')
    df['mes'] = df['fecha_y_hora_de_pedido'].dt.month
    return df

def obtener_top_3_por_mes(df):
    df = preparar_datos_historicos(df)
    # Agrupar por mes y producto
    df_mensual = df.groupby(['mes_año', 'descripcion'])['cantidad'].sum().reset_index()
    # Ordenar por fecha y luego por cantidad descendente
    df_ordenado = df_mensual.sort_values(['mes_año', 'cantidad'], ascending=[True, False])
    # Extraer los 3 mejores por mes
    top_3 = df_ordenado.groupby('mes_año').head(3)
    top_3['mes_año'] = top_3['mes_año'].astype(str)
    return top_3[['mes_año', 'descripcion', 'cantidad']]

def predecir_top_20_2027(df):
    df = preparar_datos_historicos(df)
    
    # 1. Preparación de datos para entrenamiento (CRISP-ML)
    df_hist = df.groupby(['codigo_de_producto', 'descripcion', 'mes'])['cantidad'].sum().reset_index()
    
    # Ingeniería de características: media de ventas histórica por producto
    media_ventas = df.groupby('codigo_de_producto')['cantidad'].mean().reset_index()
    media_ventas.rename(columns={'cantidad': 'media_historica'}, inplace=True)
    
    df_ml = pd.merge(df_hist, media_ventas, on='codigo_de_producto')
    
    X = df_ml[['mes', 'media_historica']]
    y = df_ml['cantidad']
    
    # 2. Entrenamiento del modelo XGBoost
    modelo = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    modelo.fit(X, y)
    
    # 3. Construcción del dataset futuro para 2027 (12 meses)
    productos_unicos = df[['codigo_de_producto', 'descripcion']].drop_duplicates()
    meses_2027 = list(range(1, 13))
    
    # Producto cartesiano: todos los productos para todos los meses
    futuro = productos_unicos.assign(key=1).merge(pd.DataFrame({'mes': meses_2027, 'key': 1}), on='key').drop('key', axis=1)
    futuro = pd.merge(futuro, media_ventas, on='codigo_de_producto', how='left').fillna(0)
    
    # 4. Inferencia
    X_futuro = futuro[['mes', 'media_historica']]
    futuro['prediccion_cantidad'] = modelo.predict(X_futuro)
    futuro['prediccion_cantidad'] = futuro['prediccion_cantidad'].clip(lower=0).round().astype(int)
    
    # 5. Extracción del Top 20 predictivo
    futuro_ordenado = futuro.sort_values(['mes', 'prediccion_cantidad'], ascending=[True, False])
    top_20_2027 = futuro_ordenado.groupby('mes').head(20)
    
    return top_20_2027[['mes', 'descripcion', 'prediccion_cantidad']]