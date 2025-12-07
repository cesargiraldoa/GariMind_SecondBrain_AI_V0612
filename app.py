import streamlit as st
import pandas as pd
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Gari Mind", page_icon="🧠", layout="wide")

# --- MENÚ LATERAL (Navegación Manual) ---
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro (Inicio)", "📊 Reportes Ejecutivos", "🗺️ Mapa de Datos"])
st.sidebar.divider()

# ==========================================
# PÁGINA 1: CEREBRO (INICIO)
# ==========================================
if pagina == "🧠 Cerebro (Inicio)":
    st.markdown('<div style="text-align: center; font-size: 2.5rem; color: #1E3A8A;">🧠 Gari Mind Second Brain</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #4B5563;">Asistente de Logística 4.0 & Análisis de Datos</div>', unsafe_allow_html=True)
    st.divider()

    st.write("##### 💬 Pregúntale a tus datos:")
    col_preg, col_btn = st.columns([4, 1])
    with col_preg:
        pregunta = st.text_input("Consulta:", placeholder="Ej: ¿Cuál fue el día de mayor venta?", label_visibility="collapsed")
    with col_btn:
        if st.button("Analizar", type="primary", use_container_width=True):
            with st.spinner('Procesando...'):
                time.sleep(1)
            st.success("Análisis completado")

# ==========================================
# PÁGINA 2: REPORTES EJECUTIVOS
# ==========================================
elif pagina == "📊 Reportes Ejecutivos":
    st.title("📊 Reportes de Variación")
    st.info("Vista preliminar (Sin librería Plotly para evitar errores)")
    
    # Métricas clave
    c1, c2, c3 = st.columns(3)
    c1.metric("Ventas Totales", "$120M", "+12%")
    c2.metric("Promedio Mes", "$10M", "-2%")
    c3.metric("Objetivo", "85%", "Cumplido")
    
    st.divider()
    
    # Gráfico simple nativo (No falla nunca)
    st.subheader("Tendencia de Ventas")
    datos_simulados = pd.DataFrame({
        'Mes': ['Ene', 'Feb', 'Mar', 'Abr', 'May'],
        'Ventas': [100, 120, 90, 110, 130]
    })
    st.bar_chart(datos_simulados.set_index('Mes'))

# ==========================================
# PÁGINA 3: MAPA DE DATOS (Tu código)
# ==========================================
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Explorador SQL", layout="wide")
st.title("🕵️ Explorador de Base de Datos")

try:
    conn = st.connection("sql", type="sql")
    st.info("Conectado a Dentisalud")
    
    # 1. Mapa de Tablas
    query_mapa = """
    SELECT TABLE_SCHEMA as Esquema, TABLE_NAME as Tabla 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME;
    """
    df_tablas = conn.query(query_mapa, ttl=600)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("📂 **Tablas Disponibles**")
        st.dataframe(df_tablas, use_container_width=True, height=500)

    with col2:
        st.write("🧪 **Probador de Datos**")
        lista = df_tablas["Esquema"] + "." + df_tablas["Tabla"]
        seleccion = st.selectbox("Elige una tabla:", lista)
        
        if st.button(f"Ver datos de {seleccion}"):
            try:
                # Top 50 para no saturar
                df = conn.query(f"SELECT TOP 50 * FROM {seleccion}", ttl=0)
                st.success(f"✅ Acceso correcto: {len(df)} filas recuperadas")
                st.dataframe(df)
            except Exception as e:
                st.error("⛔ Sin permiso o tabla vacía")
                st.write(e)

except Exception as e:
    st.error("Error de conexión")
    st.write(e)
