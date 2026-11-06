import streamlit as st
import pandas as pd
import re
from database import iniciar_conexion, subir_csv_a_bucket, cargar_csv_desde_bucket
from ml_pipeline import obtener_top_3_por_mes, entrenar_y_predecir, obtener_evaluacion_historica, calcular_metricas_tesis, obtener_resumen_metricas

st.set_page_config(page_title="Predicción de Inventarios", layout="wide")

supabase = iniciar_conexion()
BUCKET_NAME = "archivos-tesis" 

st.title("📦 Sistema Predictivo de Inventarios")

archivos_subidos = st.file_uploader("Sube tus archivos transaccionales (CSV)", type=["csv"], accept_multiple_files=True)

if archivos_subidos:
    if st.button("Procesar y Analizar"):
        with st.spinner("Subiendo archivos a Supabase y consolidando datos..."):
            lista_dataframes = [] 
            
            for archivo in archivos_subidos:
                bytes_data = archivo.getvalue()
                nombre_original = archivo.name
                
                nombre_seguro = re.sub(r'[^a-zA-Z0-9_.-]', '_', nombre_original)
                exito, msj, nombre_almacenado = subir_csv_a_bucket(supabase, BUCKET_NAME, bytes_data, nombre_original)
                
                if exito:
                    df_temp = cargar_csv_desde_bucket(supabase, BUCKET_NAME, nombre_almacenado)
                    if df_temp is not None:
                        lista_dataframes.append(df_temp)
                else:
                    st.error(f"Error al subir {nombre_original}: {msj}")
            
            if len(lista_dataframes) > 0:
                st.success(f"{len(lista_dataframes)} archivo(s) sincronizado(s) en la nube. Modelo entrenado con éxito.")
                
                df_consolidado = pd.concat(lista_dataframes, ignore_index=True)
                
                st.session_state['df_historico'] = obtener_top_3_por_mes(df_consolidado)
                st.session_state['df_predicciones'] = entrenar_y_predecir(df_consolidado)
                st.session_state['df_evaluacion'] = obtener_evaluacion_historica(df_consolidado)

if 'df_historico' in st.session_state and 'df_predicciones' in st.session_state:
    
    # Se añade la tercera pestaña "Resultados"
    tab1, tab2, tab3 = st.tabs(["Análisis Histórico", "Proyecciones", "Resultados"])
    
    with tab1:
        st.subheader("Top 3 Productos de mayor rotación por Mes")
        st.dataframe(st.session_state['df_historico'], hide_index=True, use_container_width=True)
        
    with tab2:
        df_pred = st.session_state['df_predicciones']
        col_anio, col_mes, col_top = st.columns(3)
        
        anio_str = col_anio.text_input("Año de Proyección", value="2027")
        anio_objetivo = int(anio_str) if anio_str.isdigit() else 2027
        
        meses_dict = {3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio', 
                      7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
        
        mes_nombre = col_mes.selectbox("Mes a evaluar", options=list(meses_dict.values()))
        mes_numero = [k for k, v in meses_dict.items() if v == mes_nombre][0]
        
        top_str = col_top.text_input("Cantidad de productos a visualizar", value="20")
        top_n = int(top_str) if top_str.isdigit() else 20
        
        todos_los_productos = ["Todos"] + sorted(df_pred['Producto'].unique().tolist())
        producto_filtro = st.selectbox("O buscar proyección de un producto específico", options=todos_los_productos)
        
        df_filtrado = df_pred[df_pred['mes'] == mes_numero].copy()
        
        if producto_filtro != "Todos":
            df_filtrado = df_filtrado[df_filtrado['Producto'] == producto_filtro]
        else:
            df_filtrado = df_filtrado.sort_values(by='Cantidad Proyectada', ascending=False).head(top_n)
        
        df_mostrar = df_filtrado[['Producto', 'Cantidad Proyectada']]
        
        st.subheader(f"Predicción Productos {anio_objetivo}")
        st.dataframe(df_mostrar, hide_index=True, use_container_width=True)
        
    # ---------------- PESTAÑA 3: RESULTADOS TESIS ----------------
    with tab3:
        df_eval = st.session_state['df_evaluacion']
        
        st.subheader("🏆 Ranking de Mejor Precisión Predictiva")
        st.markdown("Lista de productos donde el modelo logró el menor margen de error (ordenados por WAPE).")
        
        df_ranking = obtener_resumen_metricas(df_eval)
        
        top_ranking_str = st.text_input("Cantidad de mejores productos a mostrar", value="10")
        top_ranking_n = int(top_ranking_str) if top_ranking_str.isdigit() else 10
        
        st.dataframe(df_ranking.head(top_ranking_n), hide_index=True, use_container_width=True)
        
        st.divider()
