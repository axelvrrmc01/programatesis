import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def preparar_datos_historicos(df):
    df['fecha_y_hora_de_pedido'] = pd.to_datetime(df['fecha_y_hora_de_pedido'])
    df['mes_año'] = df['fecha_y_hora_de_pedido'].dt.to_period('M')
    df['mes'] = df['fecha_y_hora_de_pedido'].dt.month
    return df

def formatear_mes_anio(periodo_str):
    meses = {
        '01':'Enero', '02':'Febrero', '03':'Marzo', '04':'Abril', 
        '05':'Mayo', '06':'Junio', '07':'Julio', '08':'Agosto', 
        '09':'Septiembre', '10':'Octubre', '11':'Noviembre', '12':'Diciembre'
    }
    año, mes = periodo_str.split('-')
    return f"{meses[mes]} {año}"

def obtener_top_3_por_mes(df):
    df = preparar_datos_historicos(df)
    
    df_mensual = df.groupby(['mes_año', 'descripcion'])['cantidad'].sum().reset_index()
    df_ordenado = df_mensual.sort_values(['mes_año', 'cantidad'], ascending=[True, False])
    top_3 = df_ordenado.groupby('mes_año').head(3).copy()
    
    top_3['mes_año'] = top_3['mes_año'].astype(str)
    top_3['Periodo'] = top_3['mes_año'].apply(formatear_mes_anio)
    top_3 = top_3.rename(columns={'descripcion': 'Producto', 'cantidad': 'Cantidad'})
    
    return top_3[['Periodo', 'Producto', 'Cantidad']]

def entrenar_y_predecir(df):
    df = preparar_datos_historicos(df)
    
    df_hist = df.groupby(['codigo_de_producto', 'descripcion', 'mes'])['cantidad'].sum().reset_index()
    media_ventas = df.groupby('codigo_de_producto')['cantidad'].mean().reset_index()
    media_ventas.rename(columns={'cantidad': 'media_historica'}, inplace=True)
    
    df_ml = pd.merge(df_hist, media_ventas, on='codigo_de_producto')
    X = df_ml[['mes', 'media_historica']]
    y = df_ml['cantidad']
    
    modelo = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    modelo.fit(X, y)
    
    productos_unicos = df[['codigo_de_producto', 'descripcion']].drop_duplicates()
    meses_futuros = list(range(1, 13))
    
    futuro = productos_unicos.assign(key=1).merge(pd.DataFrame({'mes': meses_futuros, 'key': 1}), on='key').drop('key', axis=1)
    futuro = pd.merge(futuro, media_ventas, on='codigo_de_producto', how='left').fillna(0)
    
    X_futuro = futuro[['mes', 'media_historica']]
    futuro['Cantidad Proyectada'] = modelo.predict(X_futuro)
    futuro['Cantidad Proyectada'] = futuro['Cantidad Proyectada'].clip(lower=0).round().astype(int)
    futuro = futuro.rename(columns={'descripcion': 'Producto'})
    
    return futuro[['mes', 'Producto', 'Cantidad Proyectada']]

def obtener_evaluacion_historica(df):
    """Genera predicciones sobre la misma data histórica para comparar real vs proyectado"""
    df = preparar_datos_historicos(df)
    df_hist = df.groupby(['codigo_de_producto', 'descripcion', 'mes'])['cantidad'].sum().reset_index()
    media_ventas = df.groupby('codigo_de_producto')['cantidad'].mean().reset_index()
    media_ventas.rename(columns={'cantidad': 'media_historica'}, inplace=True)
    
    df_ml = pd.merge(df_hist, media_ventas, on='codigo_de_producto')
    X = df_ml[['mes', 'media_historica']]
    y = df_ml['cantidad']
    
    modelo = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    modelo.fit(X, y)
    
    df_ml['Cantidad Proyectada'] = modelo.predict(X)
    df_ml['Cantidad Proyectada'] = df_ml['Cantidad Proyectada'].clip(lower=0).round().astype(int)
    df_ml = df_ml.rename(columns={'descripcion': 'Producto', 'cantidad': 'Cantidad Real'})
    
    return df_ml[['mes', 'Producto', 'Cantidad Real', 'Cantidad Proyectada']]

def calcular_metricas_tesis(y_real, y_pred):
    """Calcula los indicadores MAE, RMSE y WAPE definidos en la matriz de consistencia"""
    if len(y_real) == 0:
        return 0, 0, 0
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    suma_real = np.sum(y_real)
    wape = np.sum(np.abs(y_real - y_pred)) / suma_real if suma_real > 0 else 0
    return round(mae, 2), round(rmse, 2), round(wape, 4)
def obtener_resumen_metricas(df_eval):
    """Calcula las métricas para todos los productos y los rankea del menor error al mayor"""
    resultados = []
    for producto in df_eval['Producto'].unique():
        df_prod = df_eval[df_eval['Producto'] == producto]
        mae, rmse, wape = calcular_metricas_tesis(df_prod['Cantidad Real'], df_prod['Cantidad Proyectada'])
        
        # Solo consideramos productos que tuvieron ventas reales (suma > 0) para evitar divisiones falsas
        if df_prod['Cantidad Real'].sum() > 0:
            resultados.append({
                'Producto': producto,
                'MAE': mae,
                'RMSE': rmse,
                'WAPE': wape
            })
    
    df_resumen = pd.DataFrame(resultados)
    
    # Ordenamos por WAPE ascendente (menor error = mejor predicción)
    df_resumen = df_resumen.sort_values(by=['WAPE', 'MAE'], ascending=[True, True])
    
    # Formateamos el WAPE como porcentaje visualmente atractivo
    df_resumen['WAPE'] = (df_resumen['WAPE'] * 100).round(2).astype(str) + ' %'
    
    return df_resumen
