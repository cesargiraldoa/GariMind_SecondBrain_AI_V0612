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
elif pagina == "🗺️ Mapa de Datos":
    st.title("🗺️ Mapa de la Base de Datos Dentisalud")
    
    # --- AQUÍ VA TU CÓDIGO DEL PROBADOR ---
    try:
        # Intento de conexión seguro
        conn = st.connection("sql", type="sql")
        
        # 1. Obtenemos las tablas
        query_mapa = """
            SELECT TABLE_SCHEMA as Esquema, TABLE_NAME as Tabla 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME;
        """
        df_tablas = conn.query(query_mapa, ttl=600)
        
        c_izq, c_der = st.columns([1, 2])
        
        with c_izq:
            st.success(f"✅ Se encontraron {len(df_tablas)} tablas.")
            st.dataframe(df_tablas, use_container_width=True)
            
        with c_der:
            st.subheader("🧪 Probador de Permisos")
            lista_tablas = df_tablas['Esquema'] + "." + df_tablas['Tabla']
            tabla_seleccionada = st.selectbox("Selecciona tabla:", lista_tablas)
            
            if st.button(f"Espiar {tabla_seleccionada}"):
                df_preview = conn.query(f"SELECT * FROM {tabla_seleccionada} LIMIT 5;", ttl=60)
                st.dataframe(df_preview)
                
    except Exception as e:
        # Si falla la conexión, mostramos el mensaje pero NO rompemos el menú
        st.warning("No se pudo conectar a la base de datos automáticamente.")
        st.error(f"Error técnico: {e}")
