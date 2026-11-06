import streamlit as st
import pandas as pd
import re # <-- Nueva importación para limpiar texto
from database import iniciar_conexion, subir_csv_a_bucket, cargar_csv_desde_bucket
from ml_pipeline import obtener_top_3_por_mes, predecir_top_20_2027

st.set_page_config(page_title="Predicción de Inventarios PYME", layout="wide")

# Inicializar Supabase
supabase = iniciar_conexion()
BUCKET_NAME = "archivos-tesis" 

st.title("📦 Sistema Predictivo de Inventarios PYME")
st.markdown("Basado en el marco de trabajo **CRISP-ML(Q)** para optimización de la cadena de suministro retail.")

archivos_subidos = st.file_uploader("Sube tus archivos transaccionales (CSV)", type=["csv"], accept_multiple_files=True)

if archivos_subidos:
    if st.button("Procesar y Analizar"):
        with st.spinner("Subiendo archivos a Supabase y consolidando datos..."):
            lista_dataframes = [] 
            
            for archivo in archivos_subidos:
                bytes_data = archivo.getvalue()
                nombre_original = archivo.name
                
                # MODIFICACIÓN: Limpiar el nombre (ej. "BD Libreria 2024 (1).csv" -> "BD_Libreria_2024__1_.csv")
                nombre_seguro = re.sub(r'[^a-zA-Z0-9_.-]', '_', nombre_original)
                
                # Usamos nombre_seguro para subirlo a Supabase
                exito, msj, nombre_seguro = subir_csv_a_bucket(supabase, BUCKET_NAME, bytes_data, nombre_original)
                df_temp = cargar_csv_desde_bucket(supabase, BUCKET_NAME, nombre_seguro)
                
                if exito:
                    # Lo descargamos usando el mismo nombre_seguro
                    df_temp = cargar_csv_desde_bucket(supabase, BUCKET_NAME, nombre_seguro)
                    if df_temp is not None:
                        lista_dataframes.append(df_temp)
                else:
                    st.error(f"Error al subir {nombre_original}: {msj}")
            
            if len(lista_dataframes) > 0:
                st.success(f"{len(lista_dataframes)} archivo(s) sincronizado(s) en la nube y listos para análisis.")
                
                df_consolidado = pd.concat(lista_dataframes, ignore_index=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Análisis Histórico: Top 3 Productos por Mes")
                    top_3_hist = obtener_top_3_por_mes(df_consolidado)
                    st.dataframe(top_3_hist, use_container_width=True)
                
                with col2:
                    st.subheader("🤖 Predicción Machine Learning: Top 20 para 2027")
                    top_20_pred = predecir_top_20_2027(df_consolidado)
                    st.dataframe(top_20_pred, use_container_width=True)