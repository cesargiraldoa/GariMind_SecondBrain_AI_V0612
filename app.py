import streamlit as st
import pandas as pd
import time
import os
from google import genai
from google.genai import types

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Gari Mind", page_icon="🧠", layout="wide")

# --- MENÚ LATERAL (Navegación Manual) ---
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro (Inicio)", "📊 Reportes Ejecutivos", "🗺️ Mapa de Datos"])
st.sidebar.divider()

# ==========================================
# PÁGINA 1: CEREBRO (INICIO) - LÓGICA DE IA FINAL
# ==========================================
if pagina == "🧠 Cerebro (Inicio)":
    import os
    st.title("Diagnóstico de API Key")
    
    # Intenta leer la clave que necesita la librería
    api_key_status = os.getenv("GEMINI_API_KEY")

    if api_key_status and len(api_key_status) > 10:
        st.success("✅ ¡CLAVE ENCONTRADA Y CONFIGURADA!")
        st.write("Ahora que el secreto está cargado, volvamos al código de la IA.")
    else:
        st.error("⛔ ERROR CRÍTICO: LA CLAVE DE API NO ESTÁ CARGADA EN EL ENTORNO.")
        st.warning("Debe verificar que la variable de entorno o el secreto de Streamlit (secrets.toml) esté nombrado **GEMINI_API_KEY** y contenga el valor correcto.")

    st.divider()
    st.code(f"Valor leído de la variable GEMINI_API_KEY: {api_key_status[:5]}...{api_key_status[-5:] if api_key_status else 'VACÍO'}")


# ==========================================
# PÁGINA 2: REPORTES EJECUTIVOS (FUNCIONAL Y CORREGIDO)
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
        
        # Procesamiento Pandas (Limpieza de datos - FIX de TypeError)
        df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y', errors='coerce')
        df.dropna(subset=['fecha'], inplace=True)
        
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce') 
        df.dropna(subset=['valor'], inplace=True) 
        
        df['mes_anio'] = df['fecha'].dt.strftime('%Y-%m')

    except Exception as e:
        st.error("⛔ Error al cargar los datos.")
        st.write(e)
        st.stop()

    # --- BARRERA DE FILTRO (DEFINICIÓN DE df_filtrado) ---
    st.sidebar.header("Filtros de Reporte")
    sucursales = ["Todas"] + list(df['sucursal'].unique())
    filtro_sucursal = st.sidebar.selectbox("Filtrar por Sucursal:", sucursales)

    df_filtrado = df.copy() 
    if filtro_sucursal != "Todas":
        df_filtrado = df[df['sucursal'] == filtro_sucursal]
    # --- FIN BARRERA DE FILTRO ---

    # --- Lógica de Variación y KPIs ---
    df_mensual = df_filtrado.groupby('mes_anio')['valor'].sum().reset_index()
    
    df_mensual['variacion_pct'] = df_mensual['valor'].pct_change() * 100
    df_mensual['variacion_pct'] = df_mensual['variacion_pct'].fillna(0)

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
                    st.balloons() 
                    st.dataframe(df)
                except Exception as e:
                    st.error("⛔ Sin permiso o tabla vacía")
                    st.write(e)

    except Exception as e:
        st.error("Error de conexión")
        st.write(e)
