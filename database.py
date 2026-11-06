import streamlit as st
from supabase import create_client, Client
import pandas as pd
import tempfile
import os
import io
import re

@st.cache_resource
def iniciar_conexion() -> Client:
    # Quitamos la barra final para evitar doble // en la URL oculta
    url = st.secrets["SUPABASE_URL"].rstrip("/") 
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

def limpiar_nombre(nombre: str) -> str:
    # Reemplaza cualquier cosa que no sea letra, número, punto o guion por un guion bajo '_'
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', nombre)

def subir_csv_a_bucket(supabase: Client, nombre_bucket: str, archivo_bytes: bytes, nombre_destino: str):
    ruta_temporal = None
    try:
        # 1. Sanitizar el nombre obligatoriamente (ej. "BD Libreria (1).csv" -> "BD_Libreria__1_.csv")
        nombre_seguro = limpiar_nombre(nombre_destino)

        # 2. Crear archivo local temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(archivo_bytes)
            ruta_temporal = tmp.name

        # 3. Subir a Supabase usando el nombre seguro
        supabase.storage.from_(nombre_bucket).upload(
            path=nombre_seguro,         
            file=ruta_temporal,          
            file_options={"content-type": "text/csv", "upsert": "true"}
        )
        
        # 4. Borrar archivo temporal
        os.remove(ruta_temporal)
        
        return True, "Archivo subido correctamente.", nombre_seguro
        
    except Exception as e:
        if ruta_temporal and os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)
        return False, f"{str(e)}", nombre_destino

def cargar_csv_desde_bucket(supabase: Client, nombre_bucket: str, nombre_archivo: str) -> pd.DataFrame:
    try:
        # Aseguramos descargar usando el nombre ya limpio
        nombre_seguro = limpiar_nombre(nombre_archivo)
        respuesta_bytes = supabase.storage.from_(nombre_bucket).download(nombre_seguro)
        df = pd.read_csv(io.BytesIO(respuesta_bytes))
        return df
    except Exception as e:
        st.error(f"Error al descargar {nombre_archivo}: {str(e)}")
        return None