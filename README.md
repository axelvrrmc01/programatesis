# Sistema Predictivo de Inventarios PYME

Aplicación web desarrollada en Python (Streamlit) y conectada a Supabase para analizar transacciones históricas y proyectar la demanda de inventarios mediante algoritmos de Machine Learning (XGBoost), aplicando la metodología CRISP-ML(Q).

## Instalación y Ejecución

1. Clonar el repositorio.
2. Crear un entorno virtual: `python -m venv venv` y activarlo.
3. Instalar dependencias: `pip install -r requirements.txt`.
4. Crear la carpeta `.streamlit` en la raíz y dentro un archivo `secrets.toml` con tus credenciales de Supabase.
5. Ejecutar la aplicación: `streamlit run app.py`.

## Estructura
- `app.py`: Interfaz de usuario.
- `database.py`: Control de almacenamiento en la nube.
- `ml_pipeline.py`: Núcleo de limpieza, ingeniería de características y modelado predictivo.