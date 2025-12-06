import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuración de la página
st.set_page_config(
    page_title="Gari Data Science Lab",
    page_icon="🧬",
    layout="wide"
)

# 2. Título y Descripción
st.title("🧬 Laboratorio de Ciencia de Datos & IA")
st.markdown("""
Bienvenido al entorno de análisis. 
Sube tus datos (Excel o CSV) para comenzar el análisis exploratorio y razonamiento con IA.
""")

# 3. Sidebar (Barra lateral para configuración y carga)
with st.sidebar:
    st.header("1. Carga de Datos")
    uploaded_file = st.file_uploader("Sube tu archivo aquí", type=["csv", "xlsx"])
    
    st.divider()
    
    st.header("2. Configuración IA")
    api_key = st.text_input("Gemini API Key", type="password", help="Pega tu API Key aquí o configurala en Secrets")

# 4. Lógica Principal
if uploaded_file is not None:
    # Detectar tipo de archivo y cargar
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Guardar en sesión para persistencia
        st.session_state['df'] = df
        
        st.success(f"¡Archivo '{uploaded_file.name}' cargado con éxito!")
        
        # 5. Vista Previa de Datos
        st.subheader("📊 Vista Previa de Datos")
        st.dataframe(df.head())
        
        # 6. Estadísticas Básicas Automáticas
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Filas", df.shape[0])
        col1.metric("Total Columnas", df.shape[1])
        col2.metric("Variables Numéricas", len(df.select_dtypes(include=['number']).columns))
        col3.metric("Variables Texto", len(df.select_dtypes(include=['object']).columns))

        # 7. Área de Gráficos Rápidos (Ejemplo con Plotly)
        st.subheader("📈 Visualización Rápida")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            x_axis = st.selectbox("Eje X", df.columns, index=0)
            y_axis = st.selectbox("Eje Y", numeric_cols, index=0)
            
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} por {x_axis}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay columnas numéricas para graficar.")

    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

else:
    st.info("👈 Esperando archivo. Por favor sube un CSV o Excel en la barra lateral.")

# 8. Espacio reservado para el Chat con IA (Próxima fase)
st.divider()
st.subheader("🤖 Consultas al Motor de Razonamiento")
user_question = st.text_input("Pregúntale algo a tus datos (ej: ¿Cuál es la tendencia?)")

if user_question:
    if not api_key:
        st.warning("Necesitas ingresar tu Gemini API Key en la barra lateral para usar la IA.")
    else:
        st.write("⏳ Conectando con Gemini... (Lógica a implementar en el siguiente paso)")
