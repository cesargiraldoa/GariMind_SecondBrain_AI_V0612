import streamlit as st
import pandas as pd
import sqlite3
import os
import re
from google import genai
from google.genai import types

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Gari Mind", page_icon="🧠", layout="wide")
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro (Inicio)", "📊 Reportes Ejecutivos", "🗺️ Mapa de Datos"])
st.sidebar.divider()

# Función para cargar y LIMPIAR los datos (El Preprocesamiento que pediste)
@st.cache_data(ttl=600)
def obtener_datos_limpios():
    try:
        # 1. Conectar a SQL Server original
        conn = st.connection("sql", type="sql")
        # Traemos todo como texto para evitar errores de conversión en el servidor
        df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
        
        # 2. PREPROCESAMIENTO (La clave de la solución)
        # Forzar conversión de VALOR a NÚMERO (lo que falle se vuelve 0)
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
        
        # Forzar conversión de FECHA (dd/mm/yyyy)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return pd.DataFrame()

# ==========================================
# PÁGINA 1: CEREBRO (MOTOR DE DATOS LIMPIOS)
# ==========================================
if pagina == "🧠 Cerebro (Inicio)":
    
    # Configurar API Key
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    except:
        st.error("Falta GEMINI_API_KEY en secrets.")
        st.stop()

    st.markdown('<div style="text-align: center; font-size: 2.5rem; color: #1E3A8A;">🧠 Gari Mind Second Brain</div>', unsafe_allow_html=True)
    st.info("💡 Este sistema ahora descarga los datos, corrige los tipos (Valor a número, Fecha a fecha) y permite a la IA consultar la versión limpia.")

    # 1. Cargar y Limpiar
    df_clean = obtener_datos_limpios()
    
    if not df_clean.empty:
        # 2. Crear Base de Datos Temporal en Memoria (SQLite)
        # Esto nos permite usar SQL sobre los datos YA LIMPIOS
        conn_mem = sqlite3.connect(':memory:', check_same_thread=False)
        df_clean.to_sql('ventas', conn_mem, index=False, if_exists='replace')
        
        col_preg, col_btn = st.columns([4, 1])
        with col_preg:
            pregunta = st.text_input("Consulta:", placeholder="Ej: Comparar ingresos de Kennedy vs La Playa")
        with col_btn:
            btn = st.button("Analizar", type="primary", use_container_width=True)

        if btn and pregunta:
            with st.spinner("Analizando datos limpios..."):
                # Prompt adaptado para SQLite (el motor en memoria)
                prompt = f"""
                Eres un experto en SQL. Genera una consulta SQL (Syntax SQLite) para: "{pregunta}"
                
                Tabla: ventas
                Columnas: 
                 - Fecha (DATETIME)
                 - Valor (FLOAT) -> YA ES NUMÉRICO, PUEDES SUMAR DIRECTAMENTE.
                 - Sucursal (TEXT)
                
                IMPORTANTE:
                - SQLite usa strftime('%Y-%m', Fecha) para mes.
                - Responde SOLO el código SQL dentro de ```sql ... ```.
                """
                
                try:
                    # Generar SQL
                    res = client.models.generate_content(model='gemini-2.5-flash', contents=[prompt])
                    match = re.search(r"```sql(.*?)```", res.text, re.DOTALL)
                    
                    if match:
                        sql = match.group(1).strip()
                        st.code(sql, language="sql")
                        
                        # Ejecutar en memoria (Datos Limpios)
                        df_res = pd.read_sql_query(sql, conn_mem)
                        st.success("✅ Resultados:")
                        st.dataframe(df_result := df_res) # Asignación walrus para usar después
                        
                        # Análisis final
                        res_txt = client.models.generate_content(
                            model='gemini-2.5-flash', 
                            contents=[f"Analiza esto: {pregunta}\n\n{df_result.to_markdown()}"]
                        )
                        st.markdown(res_txt.text)
                    else:
                        st.error("No se generó SQL válido.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# PÁGINA 2: REPORTES (USANDO DATOS LIMPIOS)
# ==========================================
elif pagina == "📊 Reportes Ejecutivos":
    st.title("📊 Variación de Ingresos")
    
    df = obtener_datos_limpios() # Reutilizamos la función de limpieza
    
    if not df.empty:
        df['Mes'] = df['Fecha'].dt.strftime('%Y-%m')
        
        sucursal = st.sidebar.selectbox("Sucursal", ["Todas"] + list(df['Sucursal'].unique()))
        if sucursal != "Todas": df = df[df['Sucursal'] == sucursal]
        
        df_grp = df.groupby('Mes')['Valor'].sum().reset_index()
        df_grp['Var'] = df_grp['Valor'].pct_change().fillna(0)*100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"${df['Valor'].sum():,.0f}")
        c2.metric("Promedio", f"${df_grp['Valor'].mean():,.0f}")
        c3.metric("Última Var.", f"{df_grp['Var'].iloc[-1]:.1f}%")
        
        st.bar_chart(df_grp.set_index('Mes')['Valor'])
        with st.expander("Ver Datos"): st.dataframe(df_grp)

# ==========================================
# PÁGINA 3: MAPA (DIRECTO)
# ==========================================
elif pagina == "🗺️ Mapa de Datos":
    st.title("🗺️ Mapa de Datos")
    try:
        conn = st.connection("sql", type="sql")
        tabs = conn.query("SELECT TABLE_SCHEMA+'.'+TABLE_NAME as Tabla FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'", ttl=600)
        t = st.selectbox("Tabla:", tabs['Tabla'])
        if st.button("Ver Muestra"):
            st.dataframe(conn.query(f"SELECT TOP 50 * FROM {t}", ttl=0))
    except Exception as e:
        st.error(f"Error: {e}")
