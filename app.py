import streamlit as st
import pandas as pd
import re
from database import iniciar_conexion, subir_csv_a_bucket, cargar_csv_desde_bucket
from ml_pipeline import obtener_top_3_por_mes, entrenar_y_predecir

st.set_page_config(page_title="Predicción de Inventarios PYME", layout="wide")

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
                
                # Consolidar datos
                df_consolidado = pd.concat(lista_dataframes, ignore_index=True)
                
                # Generar DataFrames base
                st.session_state['df_historico'] = obtener_top_3_por_mes(df_consolidado)
                st.session_state['df_predicciones'] = entrenar_y_predecir(df_consolidado)

# Mostrar la interfaz si los datos ya fueron procesados y están en memoria
if 'df_historico' in st.session_state and 'df_predicciones' in st.session_state:
    
    # Crear las dos pestañas solicitadas
    tab1, tab2 = st.tabs(["Análisis Histórico", "Predicciones Machine Learning"])
    
    # ---------------- PESTAÑA 1 ----------------
    with tab1:
        st.subheader("Top 3 Productos de mayor rotación por Mes")
        # El parámetro hide_index=True elimina la fila de números a la izquierda
        st.dataframe(st.session_state['df_historico'], hide_index=True, use_container_width=True)
        
    # ---------------- PESTAÑA 2 ----------------
    with tab2:
        df_pred = st.session_state['df_predicciones']
        
        # Filtros interactivos en 3 columnas
        col_anio, col_mes, col_top = st.columns(3)
        
        anio_objetivo = col_anio.number_input("Año para la predicción", min_value=2025, max_value=2035, value=2027)
        
        meses_dict = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio', 
                      7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
        
        mes_nombre = col_mes.selectbox("Mes a evaluar", options=list(meses_dict.values()))
        mes_numero = [k for k, v in meses_dict.items() if v == mes_nombre][0]
        
        top_n = col_top.number_input("Cantidad de productos a visualizar", min_value=1, max_value=100, value=20)
        
        # Búsqueda específica por producto
        todos_los_productos = ["Todos"] + sorted(df_pred['Producto'].unique().tolist())
        producto_filtro = st.selectbox("O buscar proyección de un producto específico", options=todos_los_productos)
        
        # Lógica de filtrado
        df_filtrado = df_pred[df_pred['mes'] == mes_numero].copy()
        
        if producto_filtro != "Todos":
            df_filtrado = df_filtrado[df_filtrado['Producto'] == producto_filtro]
        else:
            df_filtrado = df_filtrado.sort_values(by='Cantidad Proyectada', ascending=False).head(top_n)
        
        # Eliminar la columna mes (ya no es necesaria) y estructurar vista final
        df_mostrar = df_filtrado[['Producto', 'Cantidad Proyectada']]
        
        st.subheader(f"Predicción Productos {anio_objetivo}")
        st.dataframe(df_mostrar, hide_index=True, use_container_width=True)
