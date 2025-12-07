import streamlit as st
import pandas as pd
import time
# Nota: La librería 'plotly' fue omitida para evitar errores. 

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
    st.markdown('<div style="text-align: center; color: #4B5563;">Asistente de Logística & Análisis de Datos</div>', unsafe_allow_html=True)
    st.divider()

    st.write("##### 💬 Pregúntale a tus datos:")
    col_preg, col_btn = st.columns([4, 1])
    with col_preg:
        pregunta = st.text_input("Consulta:", placeholder="Ej: ¿Cuál fue el día de mayor venta?", label_visibility="collapsed")
    with col_btn:
        if st.button("Analizar", type="primary", use_container_width=True):
            with st.spinner('Procesando...'):
                time.sleep(1)
            st.success("✅ Análisis Completado")
            # Aquí irá la lógica de LLM y el gráfico generado

# ==========================================
# PÁGINA 2: REPORTES EJECUTIVOS (CORREGIDO: TypeError)
# ==========================================
elif pagina == "📊 Reportes Ejecutivos":
    st.title("📊 Reporte de Variación de Ingresos")
    st.info("Reporte basado en la tabla 'stg.Ingresos_Detallados'.")

    # --- Conexión y Query SQL ---
    try:
        conn = st.connection("sql", type="sql")
        
        query = """
            SELECT 
                Fecha as fecha, 
                Valor as valor,
                Sucursal as sucursal
            FROM stg.Ingresos_Detallados
            ORDER BY Fecha
        """
        
        df = conn.query(query, ttl=600)
        
        # Procesamiento Pandas (Limpieza de datos)
        # 1. Limpieza de Fecha
        df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y', errors='coerce')
        df.dropna(subset=['fecha'], inplace=True)
        
        # 2. FIX: Convertir 'valor' a numérico forzado (resuelve TypeError)
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce') 
        df.dropna(subset=['valor'], inplace=True) # Elimina filas sin valor numérico después de la conversión
        
        df['mes_anio'] = df['fecha'].dt.strftime('%Y-%m')

    except Exception as e:
        st.error("⛔ Error al cargar los datos. Verifique la conexión o el nombre de la tabla/columnas.")
        st.write(e)
        st.stop()

    # --- Lógica de Variación y KPIs ---
    df_mensual = df_filtrado.groupby('mes_anio')['valor'].sum().reset_index()
    
    # --- ERROR SOLUCIONADO ---
    df_mensual['variacion_pct'] = df_mensual['valor'].pct_change() * 100
    # --- FIN DEL ERROR SOLUCIONADO ---

    df_mensual['variacion_pct'] = df_mensual['variacion_pct'].fillna(0)

    # --- Métricas y Gráficos (El resto del código permanece igual) ---
    total_ventas = df_filtrado['valor'].sum()
    promedio_mensual = df_mensual['valor'].mean()
    ultima_variacion = df_mensual['variacion_pct'].iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Ingresos Totales", f"${total_ventas:,.0f}")
    col2.metric("Promedio Mensual", f"${promedio_mensual:,.0f}")
    col3.metric("Variación Último Mes", f"{ultima_variacion:.1f}%", delta=f"{ultima_variacion:.1f}%")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Tendencia de Ingresos ($)")
        st.bar_chart(df_mensual.set_index('mes_anio')['valor'])

    with c2:
        st.subheader("Variación Porcentual (%)")
        st.bar_chart(df_mensual.set_index('mes_anio')['variacion_pct'])

    with st.expander("Ver tabla de datos detallada"):
        st.dataframe(df_mensual)

# ==========================================
# PÁGINA 3: MAPA DE DATOS (FUNCIONAL)
# ==========================================
elif pagina == "🗺️ Mapa de Datos":
    st.title("🗺️ Mapa de la Base de Datos Dentisalud")
    st.subheader("🕵️ Explorador de Base de Datos")

    try:
        conn = st.connection("sql", type="sql")
        
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
                    df = conn.query(f"SELECT TOP 50 * FROM {seleccion}", ttl=0)
                    st.success(f"✅ Acceso correcto: {len(df)} filas recuperadas")
                    st.balloons() # ¡Celebración activada!
                    st.dataframe(df)
                except Exception as e:
                    st.error("⛔ Sin permiso o tabla vacía")
                    st.write(e)

    except Exception as e:
        st.error("Error de conexión")
        st.write(e)
