import streamlit as st
import pandas as pd
import sqlite3
import re
import os
from google import genai

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Gari Mind Final", page_icon="🧠", layout="wide")
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro", "📊 Reportes", "🗺️ Mapa"])

# --- FUNCIÓN DE CARGA SEGURA ---
@st.cache_data(ttl=600)
def cargar_datos_seguros():
    try:
        # 1. Intentamos conectar
        conn = st.connection("sql", type="sql")
        # 2. Traemos los datos crudos
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # 3. LIMPIEZA BLINDADA
        # Convertimos Valor a número, forzando errores a 0
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        # Convertimos Fecha
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        return df
    except Exception as e:
        # Si falla SQL, devolvemos dataframe vacío para no romper la app
        return pd.DataFrame()

# ==========================================
# PÁGINA 1: CEREBRO
# ==========================================
if pagina == "🧠 Cerebro":
    st.header("🧠 Cerebro (Versión Estable)")

    # 1. GESTIÓN DE API KEY (INTELIGENTE)
    # Buscamos la clave en st.secrets (prioridad) o en variables de entorno
    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        st.error("⛔ ERROR CRÍTICO: No se encontró la GEMINI_API_KEY en los Secrets.")
        st.stop()

    client = genai.Client(api_key=api_key)

    # 2. CARGAR Y MOSTRAR ESTADO DE DATOS
    df = cargar_datos_seguros()
    if df.empty:
        st.error("No se pudieron cargar datos de la base de datos SQL Server.")
        st.stop()
    
    # Creamos el motor local (SQLite)
    conn_mem = sqlite3.connect(':memory:', check_same_thread=False)
    df.to_sql('ventas', conn_mem, index=False, if_exists='replace')
    
    # Input
    pregunta = st.text_input("Pregunta:", value="Cual fue la venta total de la sucursal Kennedy?")
    
    if st.button("Analizar"):
        status_box = st.empty()
        status_box.info("⏳ Consultando a la IA...")
        
        # --- PASO 1: OBTENER SQL ---
        prompt_sql = f"""
        Genera una consulta SQL (compatible con SQLite) para responder: "{pregunta}"
        
        Tabla: ventas
        Columnas disponibles: 
          - Fecha (datetime)
          - Valor (float) -> YA ES NÚMERO
          - Sucursal (text)
          
        REGLA: Tu respuesta debe ser ÚNICAMENTE el código SQL. No uses bloques markdown. Solo el texto de la consulta.
        Ejemplo: SELECT sum(Valor) FROM ventas
        """
        
        try:
            # Llamada simplificada (solo texto)
            res = client.models.generate_content(model='gemini-2.5-flash', contents=prompt_sql)
            
            # Limpieza básica de la respuesta (quitar comillas si las pone)
            sql_generado = res.text.replace("```sql", "").replace("```", "").strip()
            
            status_box.info(f"⚡ Ejecutando SQL: {sql_generado}")
            
            # --- PASO 2: EJECUTAR SQL LOCAL ---
            try:
                df_res = pd.read_sql_query(sql_generado, conn_mem)
                
                # MOSTRAR RESULTADOS
                st.subheader("📊 Datos Encontrados:")
                st.dataframe(df_res)
                
                # --- PASO 3: INTERPRETAR ---
                status_box.info("🤖 Generando explicación...")
                prompt_analisis = f"Explica estos datos de forma ejecutiva brevemente: {pregunta}\n\n{df_res.to_markdown()}"
                res_analisis = client.models.generate_content(model='gemini-2.5-flash', contents=prompt_analisis)
                
                st.subheader("📝 Análisis:")
                st.write(res_analisis.text)
                
                status_box.success("✅ ¡Proceso completado con éxito!")
                
            except Exception as e_sql:
                st.error(f"Error ejecutando el SQL generado: {e_sql}")
                st.warning(f"La IA generó esto: {sql_generado}")
                
        except Exception as e_api:
            st.error(f"Error conectando con Gemini: {e_api}")

# ==========================================
# PÁGINA 2: REPORTES
# ==========================================
elif pagina == "📊 Reportes":
    st.title("📊 Reportes Ejecutivos")
    df = cargar_datos_seguros()
    
    if not df.empty:
        df['Mes'] = df['Fecha'].dt.strftime('%Y-%m')
        
        # Filtros
        sucursal = st.sidebar.selectbox("Filtrar Sucursal", ["Todas"] + list(df['Sucursal'].unique()))
        if sucursal != "Todas":
            df = df[df['Sucursal'] == sucursal]
        
        # Métricas
        total = df['Valor'].sum()
        promedio = df['Valor'].mean()
        
        c1, c2 = st.columns(2)
        c1.metric("Ingreso Total", f"${total:,.0f}")
        c2.metric("Ticket Promedio", f"${promedio:,.0f}")
        
        # Gráficos
        st.subheader("Tendencia Mensual")
        df_g = df.groupby('Mes')['Valor'].sum()
        st.bar_chart(df_g)
        
        with st.expander("Ver Datos Crudos Limpios"):
            st.dataframe(df)

# ==========================================
# PÁGINA 3: MAPA
# ==========================================
elif pagina == "🗺️ Mapa":
    st.title("🗺️ Mapa de Datos")
    try:
        conn = st.connection("sql", type="sql")
        tabs = conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'", ttl=600)
        t = st.selectbox("Tabla:", tabs['TABLE_NAME'])
        if st.button("Ver Muestra"):
            st.dataframe(conn.query(f"SELECT TOP 10 * FROM {t}", ttl=0))
    except Exception as e:
        st.error(f"Error conexión: {e}")
