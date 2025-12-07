import streamlit as st
import pandas as pd
import sqlite3 # Base de datos local para datos limpios
import os
import re
from google import genai
from google.genai import types

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Gari Mind", page_icon="🧠", layout="wide")

# --- MENÚ LATERAL ---
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["🧠 Cerebro (Inicio)", "📊 Reportes Ejecutivos", "🗺️ Mapa de Datos"])
st.sidebar.divider()

# ==========================================
# PÁGINA 1: CEREBRO (ESTRATEGIA MOTOR LOCAL)
# ==========================================
if pagina == "🧠 Cerebro (Inicio)":
    
    # Configuración API
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    except Exception as e:
        st.error(f"⛔ Error API: {e}")
        st.stop()

    st.markdown('<div style="text-align: center; font-size: 2.5rem; color: #1E3A8A;">🧠 Gari Mind Second Brain</div>', unsafe_allow_html=True)
    st.divider()

    # --- CARGA Y PREPROCESAMIENTO DE DATOS (LA SOLUCIÓN DEFINITIVA) ---
    @st.cache_data(ttl=600) # Guardamos esto en memoria para no recargar a cada rato
    def cargar_datos_limpios():
        try:
            # 1. Traemos los datos crudos del SQL Server
            conn = st.connection("sql", type="sql")
            df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=0)
            
            # 2. PREPROCESAMIENTO EN PYTHON (Aquí arreglamos el nvarchar)
            # Forzamos conversión a número. Lo que no sea número se vuelve 0.
            df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
            # Forzamos conversión a fecha.
            df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
            
            return df
        except Exception as e:
            return None

    # Cargamos los datos limpios
    df_clean = cargar_datos_limpios()

    if df_clean is None:
        st.error("Error conectando a la base de datos original.")
        st.stop()

    # 3. CREAMOS UN MOTOR SQL LOCAL (SQLite) CON LOS DATOS LIMPIOS
    # Esto permite que la IA haga SQL sobre datos perfectos
    conn_mem = sqlite3.connect(':memory:', check_same_thread=False)
    df_clean.to_sql('ingresos', conn_mem, index=False, if_exists='replace')

    # --- INTERFAZ ---
    col_preg, col_btn = st.columns([4, 1])
    with col_preg:
        pregunta_usuario = st.text_input("Consulta:", placeholder="Ej: ¿Cuál fue el día de mayor venta?", label_visibility="collapsed")
    with col_btn:
        boton_analizar = st.button("Analizar", type="primary", use_container_width=True)

    if boton_analizar and pregunta_usuario:
        
        with st.spinner('Analizando datos limpios...'):
            # Prompt para la IA (Ahora consulta SQLite, que es más simple)
            sql_prompt = f"""
            Genera una consulta SQL (compatible con SQLite) para responder: "{pregunta_usuario}"
            
            TABLA: ingresos
            COLUMNAS: 
             - Fecha (DATETIME)
             - Valor (FLOAT) -> YA ES NÚMERO, PUEDES SUMAR DIRECTAMENTE.
             - Sucursal (TEXT)
            
            IMPORTANTE:
            - Para fechas usa strftime si necesitas mes/año (ej: strftime('%Y-%m', Fecha))
            - Responde SOLO con el código SQL dentro de ```sql ... ```.
            """
            
            try:
                # 1. Generar SQL
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[sql_prompt]
                )
                
                # Extraer SQL
                match = re.search(r"```sql(.*?)```", response.text, re.DOTALL)
                if match:
                    sql_query = match.group(1).strip()
                    st.code(sql_query, language="sql")
                    
                    # 2. Ejecutar en el MOTOR LOCAL (No en el servidor sucio)
                    df_result = pd.read_sql_query(sql_query, conn_mem)
                    
                    st.success("✅ Resultados:")
                    st.dataframe(df_result)
                    
                    # 3. Análisis
                    analysis_prompt = f"Analiza estos datos brevemente: {pregunta_usuario}\n\n{df_result.to_markdown()}"
                    res_analysis = client.models.generate_content(model='gemini-2.5-flash', contents=[analysis_prompt])
                    st.markdown(res_analysis.text)
                else:
                    st.error("No se generó SQL válido.")
                    st.write(response.text)
                    
            except Exception as e:
                st.error(f"Error en análisis: {e}")


# ==========================================
# PÁGINA 2: REPORTES (CON DATOS LIMPIOS)
# ==========================================
elif pagina == "📊 Reportes Ejecutivos":
    st.title("📊 Variación de Ingresos")
    
    # Reusamos la lógica de limpieza para los gráficos también
    conn = st.connection("sql", type="sql")
    df = conn.query("SELECT * FROM stg.Ingresos_Detallados", ttl=600)
    
    # Preprocesamiento
    df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Fecha'], inplace=True)
    df['Mes'] = df['Fecha'].dt.strftime('%Y-%m')
    
    # Filtros
    sucursales = ["Todas"] + list(df['Sucursal'].unique())
    filtro = st.sidebar.selectbox("Sucursal:", sucursales)
    
    if filtro != "Todas":
        df = df[df['Sucursal'] == filtro]
        
    # KPI
    total = df['Valor'].sum()
    
    df_grp = df.groupby('Mes')['Valor'].sum().reset_index()
    df_grp['Var'] = df_grp['Valor'].pct_change().fillna(0)*100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", f"${total:,.0f}")
    c2.metric("Promedio", f"${df_grp['Valor'].mean():,.0f}")
    c3.metric("Última Var.", f"{df_grp['Var'].iloc[-1]:.1f}%")
    
    st.bar_chart(df_grp.set_index('Mes')['Valor'])
    with st.expander("Ver Datos"): st.dataframe(df_grp)

# ==========================================
# PÁGINA 3: MAPA (ESTABLE)
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
