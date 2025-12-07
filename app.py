import streamlit as st

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Gari Mind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" # La barra lateral inicia abierta
)

st.sidebar.success("✅ Menú activado")
st.sidebar.write("---")

# --- Estilos CSS personalizados (Opcional: para darle estética moderna) ---
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; text-align: center;}
    .sub-header {font-size: 1.2rem; color: #4B5563; text-align: center;}
    </style>
""", unsafe_allow_html=True)

# --- Encabezado ---
st.markdown('<div class="main-header">🧠 Gari Mind Second Brain</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Asistente de Logística 4.0 & Análisis de Datos</div>', unsafe_allow_html=True)
st.divider()

# --- Área de Interacción (La "Caja Mágica") ---
col1, col2, col3 = st.columns([1, 2, 1]) # Centramos el input

with col2:
    st.write("##### 💬 Pregúntale a tus datos:")
    pregunta_usuario = st.text_input(
        "Ej: ¿Cuál fue la variación de ventas en la zona norte?",
        placeholder="Escribe tu pregunta estratégica aquí..."
    )
    
    boton_consultar = st.button("Analizar con IA", type="primary", use_container_width=True)

# --- Lógica de Respuesta (Simulación para probar diseño) ---
if boton_consultar and pregunta_usuario:
    with st.spinner('Conectando neuronas... procesando datos logísticos...'):
        # Aquí luego conectaremos tu lógica real
        import time
        time.sleep(1.5) # Simula tiempo de "pensar"
    
    # Contenedor de Resultados
    st.success("✅ Análisis Completado")
    
    # Dividimos la pantalla: Gráfico a la izquierda, Explicación a la derecha
    c_graf, c_texto = st.columns([1.5, 1])
    
    with c_graf:
        st.info("📊 [Aquí aparecerá el Gráfico Excepcional generado por IA]")
        # Placeholder para cuando metamos Plotly
        st.bar_chart({"Ene": 100, "Feb": 120, "Mar": 90}) 
        
    with c_texto:
        st.subheader("📝 Insights Ejecutivos")
        st.write("""
        **Respuesta:** Se observa una variación positiva del 20% en febrero, seguida de una caída en marzo.
        
        **Causa Raíz:** Posible desabastecimiento en la segunda semana de marzo.
        
        **Recomendación:** Revisar stock de seguridad para el Q2.
        """)
