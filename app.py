import streamlit as st
import time

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Gari Mind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Encabezado ---
st.markdown('<div style="text-align: center; font-size: 2rem; color: #1E3A8A;">🧠 Gari Mind Second Brain</div>', unsafe_allow_html=True)
st.divider()

# --- Área de Interacción ---
st.write("##### 💬 Pregúntale a tus datos:")
pregunta_usuario = st.text_input("Consulta:", placeholder="Ej: ¿Cuál fue la variación de ventas?")
    
if st.button("Analizar con IA", type="primary"):
    with st.spinner('Procesando...'):
        time.sleep(1)
    st.success("✅ Análisis Completado")
    st.info("Nota: Los gráficos avanzados se activarán cuando instalemos la librería gráfica.")

# --- MENSAJE DE DIAGNÓSTICO ---
st.sidebar.success("✅ ¡Menú Cargado!")
st.sidebar.info("Si ves esto, la estructura de carpetas funciona.")
