import streamlit as st
import pandas as pd
import sqlite3
import os
import re
from google import genai

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Gari Mind Debug", page_icon="🔧", layout="wide")
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro", "📊 Reportes", "🗺️ Mapa"])

# --- FUNCIÓN DE CARGA DE DATOS (ESTRATEGIA LOCAL) ---
@st.cache_data(ttl=600)
def cargar_datos():
    # 1. Conexión SQL Server
    conn = st.connection("sql", type="sql")
    # Traemos todo
    df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
    
    # 2. LIMPIEZA FORZADA EN PYTHON
    # Convertir Valor a número (lo que falle se vuelve 0)
    df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
    # Convertir Fecha
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
    
    return df

# ==========================================
# PÁGINA 1: CEREBRO (MODO DEBUG)
# ==========================================
if pagina == "🧠 Cerebro":
    st.title("🔧 Cerebro (Modo Diagnóstico)")
    
    # Verificación de API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ NO se detectó GEMINI_API_KEY en los secrets.")
        st.stop()
    
    client = genai.Client(api_key=api_key)

    # 1. Cargar Datos y Mostrar Estado
    with st.status("Cargando y limpiando datos...", expanded=True) as status:
        df_clean = cargar_datos()
        st.write(f"✅ Datos cargados: {len(df_clean)} filas.")
        st.write("Muestra de datos limpios (verificar columna Valor):")
        st.dataframe(df_clean[['Fecha', 'Valor', 'Sucursal']].head(3))
        
        # Crear Motor SQL en Memoria
        conn_mem = sqlite3.connect(':memory:', check_same_thread=False)
        df_clean.to_sql('ingresos', conn_mem, index=False, if_exists='replace')
        status.update(label="Datos listos en memoria RAM", state="complete", expanded=False)

    pregunta = st.text_input("Pregunta:", "Comparar ingresos de Kennedy vs La Playa")
    
    if st.button("Analizar"):
        st.write("---")
        st.info("1. Enviando pregunta a la IA...")
        
        # PROMPT
        prompt = f"""
        Genera una consulta SQL (SQLite) para: "{pregunta}"
        Tabla: ingresos
        Columnas: Fecha (datetime), Valor (float), Sucursal (text).
        
        REGLA: Responde SOLO el código SQL dentro de tres comillas invertidas ```sql ... ```
        """
        
        # LLAMADA API
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[prompt]
        )
        
        # DEBUG: MOSTRAR RESPUESTA CRUDA
        st.warning("🔍 RESPUESTA CRUDA DE GEMINI (Lo que llega realmente):")
        st.code(response.text)
        
        # EXTRACCIÓN
        match = re.search(r"```sql(.*?)```", response.text, re.DOTALL)
        if match:
            sql = match.group(1).strip()
            st.success("2. SQL Extraído:")
            st.code(sql, language="sql")
            
            # EJECUCIÓN
            try:
                df_res = pd.read_sql_query(sql, conn_mem)
                st.success(f"3. Datos obtenidos ({len(df_res)} filas):")
                st.dataframe(df_res)
                
                # ANÁLISIS
                st.info("4. Generando análisis de texto...")
                res_txt = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[f"Analiza esto: {pregunta}\nDatos:\n{df_res.to_markdown()}"]
                )
                st.markdown("### 🤖 Respuesta Final:")
                st.markdown(res_txt.text)
                
            except Exception as e:
                st.error(f"❌ Error ejecutando el SQL generado: {e}")
        else:
            st.error("❌ No se encontró el bloque ```sql ... ``` en la respuesta de arriba.")

# ==========================================
# PÁGINA 2: REPORTES
# ==========================================
elif pagina == "📊 Reportes":
    st.title("📊 Reportes")
    df = cargar_datos()
    df['Mes'] = df['Fecha'].dt.strftime('%Y-%m')
    
    sucursal = st.sidebar.selectbox("Filtro Sucursal", ["Todas"] + list(df['Sucursal'].unique()))
    if sucursal != "Todas":
        df = df[df['Sucursal'] == sucursal]
        
    st.metric("Total Ingresos", f"${df['Valor'].sum():,.0f}")
    st.bar_chart(df.groupby('Mes')['Valor'].sum())

# ==========================================
# PÁGINA 3: MAPA
# ==========================================
elif pagina == "🗺️ Mapa":
    st.title("🗺️ Mapa SQL")
    conn = st.connection("sql", type="sql")
    tabs = conn.query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
    st.dataframe(tabs)
